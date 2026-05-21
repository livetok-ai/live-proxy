let objects_handler = null;
let sentiment_handler = null;

function setup(connection) {
    if (!connection.get_model("yolo")) {
        connection.add_model("yolo");
    }
    if (!connection.get_model("face_landmarker")) {
        connection.add_model("face_landmarker");
    }

    const yolo = connection.get_model("yolo");
    const landmark = connection.get_model("face_landmarker");
    
    if (yolo && landmark) {
        logger.log_info("[Counter Script] YOLO and Face Landmarker models found on connection " + connection.id);
        
        // Initialize default state: YOLO processing & video output active; Landmarker fully inactive
        yolo.enable_input();
        yolo.enable_output();
        landmark.disable_input();
        landmark.disable_output();

        // Initialize maps to keep count of objects and sentiments
        connection.object_counts = {};
        connection.sentiment_counts = {};
        
        objects_handler = function(objects) {
            // Objects detected keep a count in a map
            for (const obj of objects) {
                connection.object_counts[obj] = (connection.object_counts[obj] || 0) + 1;
            }
            
            // If objects detected include person then disable frame generation from yolo and enable frame generation from landmarker
            if (objects.includes("person")) {
                logger.log_info("[Counter Script] Person detected. Enabling Landmarker (input + output) and disabling YOLO frame generation.");
                yolo.disable_output();
                landmark.enable_input();
                landmark.enable_output();
            } else {
                logger.log_info("[Counter Script] No person detected. Enabling YOLO frame generation and disabling Landmarker.");
                yolo.enable_output();
                landmark.disable_input();
                landmark.disable_output();
            }
        };
                
        sentiment_handler = function(sentiment) {
            // Sentiment detected keeps a count in a map
            connection.sentiment_counts[sentiment] = (connection.sentiment_counts[sentiment] || 0) + 1;
            logger.log_info("[Counter Script] Sentiment detected: " + sentiment + " (Total: " + connection.sentiment_counts[sentiment] + ")");
        };
            
        yolo.on("objects", objects_handler);
        landmark.on("sentiment", sentiment_handler);
    } else {
        logger.log_info("[Counter Script] YOLO or Face Landmarker model not found on connection " + connection.id);
    }
}

function teardown(connection) {
    // Log all counters
    const object_counts = connection.object_counts;
    const sentiment_counts = connection.sentiment_counts;
    
    if (object_counts !== undefined || sentiment_counts !== undefined) {
        logger.log_info("[Counter Script] Teardown for connection " + connection.id);
        logger.log_info("[Counter Script] Final object counts: " + JSON.stringify(object_counts));
        logger.log_info("[Counter Script] Final sentiment counts: " + JSON.stringify(sentiment_counts));
    }
    
    // Clean up event handlers
    const yolo = connection.get_model("yolo");
    const landmark = connection.get_model("face_landmarker");
    if (yolo && objects_handler) {
        yolo.off("objects", objects_handler);
    }
    if (landmark && sentiment_handler) {
        landmark.off("sentiment", sentiment_handler);
    }
}
