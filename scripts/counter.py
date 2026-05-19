import logger

def setup(connection):
    yolo = connection.get_model("yolo")
    landmark = connection.get_model("face_sentiment")
    
    if yolo and landmark:
        logger.log_info(f"[Counter Script] YOLO and Face Sentiment models found on connection {connection.id}")
        
        # Initialize default state: YOLO processing & video output active; Landmarker fully inactive
        yolo.enable_input()
        yolo.enable_output()
        landmark.disable_input()
        landmark.disable_output()

        # Initialize maps to keep count of objects and sentiments
        connection.object_counts = {}
        connection.sentiment_counts = {}
        
        def objects_detected(objects):
            # Objects detected keep a count in a map
            for obj in objects:
                connection.object_counts[obj] = connection.object_counts.get(obj, 0) + 1
            
            # If objects detected include person then disable frame generation from yolo and enable frame generation from landmarker
            if "person" in objects:
                logger.log_info("[Counter Script] Person detected. Enabling Landmarker (input + output) and disabling YOLO frame generation.")
                yolo.disable_output()
                landmark.enable_input()
                landmark.enable_output()
            else:
                logger.log_info("[Counter Script] No person detected. Enabling YOLO frame generation and disabling Landmarker.")
                yolo.enable_output()
                landmark.disable_input()
                landmark.disable_output()
                
        def sentiment_detected(sentiment):
            # Sentiment detected keeps a count in a map
            connection.sentiment_counts[sentiment] = connection.sentiment_counts.get(sentiment, 0) + 1
            logger.log_info(f"[Counter Script] Sentiment detected: {sentiment} (Total: {connection.sentiment_counts[sentiment]})")
            
        yolo.on("objects", objects_detected)
        landmark.on("sentiment", sentiment_detected)
        
        # Save handlers for clean unregistration on teardown
        connection._objects_handler = objects_detected
        connection._sentiment_handler = sentiment_detected
    else:
        logger.log_info(f"[Counter Script] YOLO or Face Sentiment model not found on connection {connection.id}")

def teardown(connection):
    # Log all counters
    object_counts = getattr(connection, "object_counts", None)
    sentiment_counts = getattr(connection, "sentiment_counts", None)
    
    if object_counts is not None or sentiment_counts is not None:
        logger.log_info(f"[Counter Script] Teardown for connection {connection.id}")
        logger.log_info(f"[Counter Script] Final object counts: {object_counts}")
        logger.log_info(f"[Counter Script] Final sentiment counts: {sentiment_counts}")
    
    # Clean up event handlers
    yolo = connection.get_model("yolo")
    landmark = connection.get_model("face_sentiment")
    if yolo and hasattr(connection, "_objects_handler"):
        yolo.off("objects", connection._objects_handler)
    if landmark and hasattr(connection, "_sentiment_handler"):
        landmark.off("sentiment", connection._sentiment_handler)
