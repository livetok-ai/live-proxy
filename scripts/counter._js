let objects_handler = null;
let sentiment_handler = null;
let last_objects_str = null;

function setup(connection) {
    const yolo = connection.add_model("yolo");
    const landmark = connection.add_model("face_landmarker");
    const gemini = connection.get_model("gemini");

    console.log("YOLO and Face Landmarker models found on connection " + connection.id);

    // Initialize default state: Landmarker fully inactive (YOLO starts active by default)
    landmark.disable_input();
    landmark.disable_output();

    // Initialize maps to keep count of objects and sentiments
    connection.object_counts = {};
    connection.sentiment_counts = {};

    connection.start_time = Date.now();

    objects_handler = function (objects) {
        if (Date.now() - connection.start_time < 5000) {
            return;
        }

        // Objects detected keep a count in a map
        for (const obj of objects) {
            connection.object_counts[obj] = (connection.object_counts[obj] || 0) + 1;
        }

        // Inform the Gemini model about the latest objects on screen every time the objects change
        const sorted_objects = [...new Set(objects)].sort();
        const objects_str = sorted_objects.join(", ");
        if (objects_str !== last_objects_str) {
            last_objects_str = objects_str;
            if (gemini) {
                const info_msg = objects_str ? "Objects currently on screen: " + objects_str : "No objects currently on screen";
                console.log("Informing Gemini of screen objects: " + info_msg);
                gemini.send_info(info_msg);
            }
        }

        // If objects detected include person then disable frame generation from yolo and enable frame generation from landmarker
        if (objects.includes("person")) {
            console.log("Person detected. Enabling Landmarker (input + output) and disabling YOLO frame generation.");
            yolo.disable_output();
            landmark.enable_input();
            landmark.enable_output();
        } else {
            console.log("No person detected. Enabling YOLO frame generation and disabling Landmarker.");
            yolo.enable_output();
            landmark.disable_input();
            landmark.disable_output();
            connection.send_data({ display: "" });
        }
    };

    sentiment_handler = function (sentiment) {
        // Sentiment detected keeps a count in a map
        connection.sentiment_counts[sentiment] = (connection.sentiment_counts[sentiment] || 0) + 1;
        console.log("Sentiment detected: " + sentiment + " (Total: " + connection.sentiment_counts[sentiment] + ")");
        connection.send_data({ display: sentiment.toUpperCase() });
    };

    yolo.on("objects_detected", objects_handler);
    landmark.on("emotions_detected", sentiment_handler);
}
