package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/emiago/sipgo"
	"github.com/emiago/sipgo/sip"
)

func main() {
	// Define CLI flags
	var (
		sipURL      string
		username    string
		duration    int
		userAgent   string
		displayName string
		transport   string
	)

	flag.StringVar(&sipURL, "url", "", "SIP URL to call (required, format: sip:user@host:port)")
	flag.StringVar(&username, "username", "sipcli", "Username for SIP client")
	flag.IntVar(&duration, "duration", 30, "Call duration in seconds (0 for indefinite)")
	flag.StringVar(&userAgent, "user-agent", "SIPGoCLI/1.0", "User-Agent header")
	flag.StringVar(&displayName, "display-name", "SIP CLI", "Display name for caller")
	flag.StringVar(&transport, "transport", "tcp", "Transport protocol (tcp or udp)")

	flag.Parse()

	if sipURL == "" {
		fmt.Println("Error: SIP URL is required")
		flag.Usage()
		os.Exit(1)
	}

	// Add transport parameter to SIP URI if not already present
	if transport != "" && transport != "udp" {
		// Check if transport parameter is already in the URI
		if !strings.Contains(sipURL, ";transport=") {
			sipURL = sipURL + ";transport=" + transport
		}
	}

	// Parse SIP URI
	recipient := &sip.Uri{}
	if err := sip.ParseUri(sipURL, recipient); err != nil {
		log.Fatalf("Failed to parse SIP URL: %v", err)
	}

	// Create User Agent
	ua, err := sipgo.NewUA(
		sipgo.WithUserAgent(userAgent),
	)
	if err != nil {
		log.Fatalf("Failed to create UA: %v", err)
	}
	defer ua.Close()

	// Create SIP client
	client, err := sipgo.NewClient(ua)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	// Setup context with timeout if duration is specified
	ctx := context.Background()
	var cancel context.CancelFunc
	if duration > 0 {
		ctx, cancel = context.WithTimeout(ctx, time.Duration(duration)*time.Second)
		defer cancel()
	}

	// Setup signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	fmt.Printf("Initiating SIP call to: %s\n", sipURL)
	fmt.Printf("User-Agent: %s\n", userAgent)
	fmt.Printf("Username: %s\n", username)
	fmt.Printf("Display Name: %s\n", displayName)
	fmt.Printf("Transport: %s\n", transport)
	if duration > 0 {
		fmt.Printf("Duration: %d seconds\n", duration)
	} else {
		fmt.Println("Duration: indefinite (press Ctrl+C to hangup)")
	}

	// Create SDP body with WebRTC ICE credentials (required for aiortc)
	sdpBody := "v=0\r\n" +
		"o=- 123456 654321 IN IP4 127.0.0.1\r\n" +
		"s=SIP Call\r\n" +
		"c=IN IP4 127.0.0.1\r\n" +
		"t=0 0\r\n" +
		"a=ice-ufrag:abcd\r\n" +
		"a=ice-pwd:abcdefghijklmnopqrstuvwx\r\n" +
		"a=fingerprint:sha-256 00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF\r\n" +
		"m=audio 49170 RTP/AVP 0 8 97\r\n" +
		"a=rtpmap:0 PCMU/8000\r\n" +
		"a=rtpmap:8 PCMA/8000\r\n" +
		"a=rtpmap:97 opus/48000/2\r\n" +
		"a=sendrecv\r\n"

	// Create INVITE request
	req := sip.NewRequest(sip.INVITE, *recipient)
	req.AppendHeader(sip.NewHeader("User-Agent", userAgent))
	req.AppendHeader(sip.NewHeader("From", fmt.Sprintf("\"%s\" <sip:%s@localhost>", displayName, username)))
	req.AppendHeader(sip.NewHeader("To", fmt.Sprintf("<sip:%s@%s>", recipient.User, recipient.Host)))
	req.AppendHeader(sip.NewHeader("Content-Type", "application/sdp"))
	req.AppendHeader(sip.NewHeader("Content-Length", fmt.Sprintf("%d", len(sdpBody))))
	req.AppendHeader(sip.NewHeader("X-SIP-CLI", "true"))
	req.SetBody([]byte(sdpBody))

	// Send the request
	fmt.Println("\nSending INVITE request with SDP...")
	tx, err := client.TransactionRequest(ctx, req)
	if err != nil {
		log.Fatalf("Failed to send request: %v", err)
	}

	fmt.Println("Request sent successfully!")

	// Wait for response or signal
	responseChan := make(chan *sip.Response, 1)
	go func() {
		select {
		case res := <-tx.Responses():
			if res != nil {
				responseChan <- res
			}
		case <-ctx.Done():
			return
		}
	}()

	select {
	case res := <-responseChan:
		if res != nil {
			fmt.Printf("Received response: %d %s\n", res.StatusCode, res.Reason)

			// If call is accepted (2xx response), wait for duration
			if res.StatusCode >= 200 && res.StatusCode < 300 {
				fmt.Println("Call established successfully!")

				if duration > 0 {
					fmt.Printf("Waiting for %d seconds...\n", duration)
					time.Sleep(time.Duration(duration) * time.Second)
				} else {
					fmt.Println("Press Ctrl+C to end call...")
					<-sigChan
				}

				// Send BYE request to end call
				fmt.Println("Sending BYE request...")
				byeReq := sip.NewRequest(sip.BYE, *recipient)
				_, err = client.TransactionRequest(ctx, byeReq)
				if err != nil {
					log.Printf("Error sending BYE: %v", err)
				}
			}
		}
	case <-sigChan:
		fmt.Println("\nReceived interrupt signal, canceling call...")
	case <-ctx.Done():
		fmt.Println("\nCall duration expired")
	}

	fmt.Println("Call completed")
}
