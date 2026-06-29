import glob
import importlib.util
import inspect
import json
import os

import logger

try:
    import quickjs
except ImportError:
    quickjs = None

LOADED_SCRIPTS = []


class JavaScriptScript:
    def __init__(self, file_path=None, code=None, name="dynamic_script.js"):
        self.file_path = file_path
        self.__name__ = name
        if file_path:
            self.__name__ = os.path.basename(file_path)
            with open(file_path, encoding="utf-8") as f:
                self.code = f.read()
        else:
            self.code = code or ""
        self.contexts = {}  # connection.id -> (ctx, callbacks)

        # Automatically generate PascalCase script prefix based on script filename
        # e.g., "face_detection.js" -> "face_detection" -> "FaceDetection" -> "[FaceDetection Script]"
        stem = os.path.splitext(self.__name__)[0]
        pascal_name = "".join(part.capitalize() for part in stem.split("_"))
        self.script_prefix = f"[{pascal_name} Script]"

    def _setup_context(self, connection):
        if quickjs is None:
            raise RuntimeError("quickjs-ng package is not installed. Please run 'uv pip install quickjs-ng'")

        ctx = quickjs.Context()
        callbacks = {}  # callback_id -> (python_cb, model_name, event_name)

        def py_log(level, msg):
            # Clean msg by stripping the script_prefix if it starts with it
            clean_msg = msg
            if clean_msg.startswith(self.script_prefix):
                clean_msg = clean_msg[len(self.script_prefix) :].lstrip()

            if level == "error":
                logger.log_info(f"{self.script_prefix} [ERROR] {clean_msg}", context=connection.id)
            else:
                logger.log_info(f"{self.script_prefix} {clean_msg}", context=connection.id)

        ctx.add_callable("py_log", py_log)
        ctx.add_callable("py_connection_get_id", lambda: connection.id)
        ctx.add_callable("py_connection_has_model", lambda name: connection.get_model(name) is not None)

        def py_connection_add_model(name: str, kwargs_json: str = None):
            kwargs = None
            if kwargs_json:
                try:
                    kwargs = json.loads(kwargs_json)
                except Exception as e:
                    logger.log_info(f"{self.script_prefix} Error parsing kwargs JSON: {e}", context=connection.id)
            connection.add_model_sync(name, kwargs)
            return None

        ctx.add_callable("py_connection_add_model", py_connection_add_model)
        ctx.add_callable("py_connection_send_data", lambda data_str: connection._send_raw_json(data_str))

        def py_connection_close():
            import asyncio
            asyncio.create_task(connection.close())

        ctx.add_callable("py_connection_close", py_connection_close)

        def py_model_enable_input(name):
            m = connection.get_model(name)
            if m:
                m.enable_input()

        def py_model_disable_input(name):
            m = connection.get_model(name)
            if m:
                m.disable_input()

        def py_model_enable_output(name):
            m = connection.get_model(name)
            if m:
                m.enable_output()

        def py_model_disable_output(name):
            m = connection.get_model(name)
            if m:
                m.disable_output()

        def py_model_send_info(name, info):
            m = connection.get_model(name)
            if m and hasattr(m, "send_info"):
                import asyncio

                asyncio.create_task(m.send_info(info))

        ctx.add_callable("py_model_enable_input", py_model_enable_input)
        ctx.add_callable("py_model_disable_input", py_model_disable_input)
        ctx.add_callable("py_model_enable_output", py_model_enable_output)
        ctx.add_callable("py_model_disable_output", py_model_disable_output)
        ctx.add_callable("py_model_send_info", py_model_send_info)

        def py_model_send(name, input_val):
            m = connection.get_model(name)
            if m:
                import asyncio

                asyncio.create_task(m.send(input_val))

        def py_model_add_tool(model_name, tool_json, callback_id):
            m = connection.get_model(model_name)
            if m and hasattr(m, "add_tool"):
                try:
                    tool_dict = json.loads(tool_json)
                    tool_name = tool_dict.get("name")

                    param_names = []
                    parameters = tool_dict.get("parameters")
                    if isinstance(parameters, list):
                        for p in parameters:
                            if isinstance(p, dict) and "name" in p:
                                param_names.append(p["name"])
                    elif isinstance(parameters, dict):
                        if "properties" in parameters and isinstance(parameters["properties"], dict):
                            param_names.extend(parameters["properties"].keys())
                        else:
                            param_names.extend(parameters.keys())

                    logger.log_info(
                        f"{self.script_prefix} Adding tool '{tool_name}' to model '{model_name}' with parameters: {param_names}",
                        context=connection.id,
                    )

                    async def on_tool_called(args):
                        import asyncio

                        try:
                            args_json = json.dumps(args)
                            res_str = ctx.eval(
                                f"JSON.stringify(_run_tool_callback({json.dumps(callback_id)}, {args_json}))"
                            )
                            res = json.loads(res_str)
                            if res.get("async"):
                                result_id = res["resultId"]
                                while True:
                                    has_jobs = ctx.execute_pending_job()
                                    status_str = ctx.eval(
                                        f"JSON.stringify(_tool_call_results[{json.dumps(result_id)}])"
                                    )
                                    status = json.loads(status_str)
                                    if status["status"] == "resolved":
                                        ctx.eval(f"delete _tool_call_results[{json.dumps(result_id)}]")
                                        return status["value"]
                                    elif status["status"] == "rejected":
                                        ctx.eval(f"delete _tool_call_results[{json.dumps(result_id)}]")
                                        return {"error": status["value"]}
                                    await asyncio.sleep(0.001)
                            else:
                                return res.get("value")
                        except Exception as e:
                            logger.log_info(
                                f"{self.script_prefix} Error in JS tool callback: {e}", context=connection.id
                            )
                            return {"error": str(e)}

                    m.add_tool(tool_dict, on_tool_called)
                    callbacks[callback_id] = (on_tool_called, model_name, "tool_call")
                except Exception as e:
                    logger.log_info(f"{self.script_prefix} Error adding tool to model: {e}", context=connection.id)

        def py_fetch(url, options_json=None):
            import json
            import urllib.error
            import urllib.request

            try:
                req = urllib.request.Request(url)
                if options_json:
                    options = json.loads(options_json)
                    if "method" in options:
                        req.method = options["method"]
                    if "headers" in options:
                        for k, v in options["headers"].items():
                            req.add_header(k, v)
                    if "body" in options:
                        body_data = options["body"]
                        if isinstance(body_data, str):
                            req.data = body_data.encode("utf-8")
                        else:
                            req.data = json.dumps(body_data).encode("utf-8")

                with urllib.request.urlopen(req, timeout=5) as response:
                    status = response.status
                    text = response.read().decode("utf-8")
                    return json.dumps({"ok": True, "status": status, "text": text})
            except urllib.error.HTTPError as e:
                return json.dumps({"ok": False, "status": e.code, "text": str(e)})
            except Exception as e:
                return json.dumps({"ok": False, "status": 500, "text": str(e)})

        ctx.add_callable("py_model_send", py_model_send)
        ctx.add_callable("py_model_add_tool", py_model_add_tool)
        ctx.add_callable("py_fetch", py_fetch)

        def py_model_on(model_name, event_name, callback_id):
            m = connection.get_model(model_name)
            if m:

                def on_event_fired(data):
                    try:
                        data_json = json.dumps(data)
                        ctx.eval(
                            f"_trigger_callback({json.dumps(model_name)}, {json.dumps(event_name)}, {json.dumps(callback_id)}, {data_json})"
                        )
                    except Exception as e:
                        logger.log_info(
                            f"{self.script_prefix} Error triggering JS callback: {e}", context=connection.id
                        )

                m.on(event_name, on_event_fired)
                callbacks[callback_id] = (on_event_fired, model_name, event_name)

        def py_model_off(model_name, event_name, callback_id):
            m = connection.get_model(model_name)
            if m and callback_id in callbacks:
                on_event_fired, _, _ = callbacks.pop(callback_id)
                m.off(event_name, on_event_fired)

        ctx.add_callable("py_model_on", py_model_on)
        ctx.add_callable("py_model_off", py_model_off)

        # Evaluate bootstrap code
        bootstrap = """
        const _callbacks = {};
        function _trigger_callback(modelName, eventType, callbackId, data) {
            const cb = _callbacks[callbackId];
            if (cb) {
                try {
                    return cb(data);
                } catch (e) {
                    py_log("error", "Error invoking JS callback: " + e + "\\n" + e.stack);
                    return { error: e.message };
                }
            }
        }
        const _tool_call_results = {};
        function _run_tool_callback(callbackId, args) {
            const cb = _callbacks[callbackId];
            if (!cb) {
                return { async: false, value: { error: "Callback not found" } };
            }
            try {
                const result = cb(args);
                if (result && typeof result.then === 'function') {
                    const resultId = Math.random().toString(36).substr(2, 9);
                    _tool_call_results[resultId] = { status: 'pending', value: null };
                    result.then(
                        v => { _tool_call_results[resultId] = { status: 'resolved', value: v }; },
                        err => { _tool_call_results[resultId] = { status: 'rejected', value: err.message || String(err) }; }
                    );
                    return { async: true, resultId: resultId };
                } else {
                    return { async: false, value: result };
                }
            } catch (e) {
                py_log("error", "Error invoking JS callback: " + e + "\\n" + e.stack);
                return { async: false, value: { error: e.message } };
            }
        }
        const logger = {
            log_info: function(msg) {
                py_log("info", msg);
            }
        };
        const console = {
            log: function(...args) {
                py_log("info", args.map(a => typeof a === 'object' ? JSON.stringify(a) : String(a)).join(' '));
            },
            error: function(...args) {
                py_log("error", args.map(a => typeof a === 'object' ? JSON.stringify(a) : String(a)).join(' '));
            }
        };
        const fetch = function(url, options) {
            try {
                const res = py_fetch(url, options ? JSON.stringify(options) : null);
                const resObj = typeof res === 'string' ? JSON.parse(res) : res;
                return {
                    ok: resObj.ok,
                    status: resObj.status,
                    text: async () => resObj.text,
                    json: async () => JSON.parse(resObj.text)
                };
            } catch (e) {
                py_log("error", "Error in fetch: " + e);
                return {
                    ok: false,
                    status: 500,
                    text: async () => String(e),
                    json: async () => ({ error: String(e) })
                };
            }
        };
        class ModelWrapper {
            constructor(name) {
                this.name = name;
            }
            enable_input() { py_model_enable_input(this.name); }
            disable_input() { py_model_disable_input(this.name); }
            enable_output() { py_model_enable_output(this.name); }
            disable_output() { py_model_disable_output(this.name); }
            send_info(info) { py_model_send_info(this.name, info); }
            send(input) { py_model_send(this.name, typeof input === 'object' ? JSON.stringify(input) : String(input)); }
            addTool(tool) {
                const callbackId = this.name + "_tool_" + tool.name + "_" + Math.random().toString(36).substr(2, 9);
                _callbacks[callbackId] = (args) => {
                    let paramValues = [];
                    if (Array.isArray(tool.parameters)) {
                        paramValues = tool.parameters.map(p => args[p.name]);
                    } else if (tool.parameters && typeof tool.parameters === 'object' && tool.parameters.properties) {
                        paramValues = Object.keys(tool.parameters.properties).map(k => args[k]);
                    } else {
                        paramValues = [args];
                    }
                    try {
                        return tool.callback(...paramValues);
                    } catch (e) {
                        py_log("error", "Error in tool callback: " + e + "\\n" + e.stack);
                        return { error: e.message };
                    }
                };
                py_model_add_tool(this.name, JSON.stringify({
                    name: tool.name,
                    description: tool.description,
                    parameters: tool.parameters
                }), callbackId);
            }
            on(event, handler) {
                const callbackId = this.name + "_" + event + "_" + Math.random().toString(36).substr(2, 9);
                _callbacks[callbackId] = handler;
                py_model_on(this.name, event, callbackId);
            }
            off(event, handler) {
                for (const [id, cb] of Object.entries(_callbacks)) {
                    if (cb === handler && id.startsWith(this.name + "_" + event + "_")) {
                        delete _callbacks[id];
                        py_model_off(this.name, event, id);
                    }
                }
            }
        }
        class ConnectionWrapper {
            constructor() {
                this.id = py_connection_get_id();
                this._models = {};
            }
            get_model(name) {
                if (!py_connection_has_model(name)) {
                    return null;
                }
                if (!this._models[name]) {
                    this._models[name] = new ModelWrapper(name);
                }
                return this._models[name];
            }
            add_model(name, kwargs) {
                py_connection_add_model(name, kwargs ? JSON.stringify(kwargs) : null);
                return this.get_model(name);
            }
            send_data(data) {
                py_connection_send_data(JSON.stringify(data));
            }
            close() {
                py_connection_close();
            }
        }
        const connection = new ConnectionWrapper();
        """
        ctx.eval(bootstrap)

        # Evaluate user script
        ctx.eval(self.code)

        self.contexts[connection.id] = (ctx, callbacks)
        return ctx

    def setup(self, connection):
        ctx = self._setup_context(connection)
        setup_func = ctx.get("setup")
        if setup_func:
            ctx.eval("setup(connection)")

    def teardown(self, connection):
        if connection.id in self.contexts:
            ctx, callbacks = self.contexts[connection.id]
            teardown_func = ctx.get("teardown")
            if teardown_func:
                try:
                    ctx.eval("teardown(connection)")
                except Exception as e:
                    logger.log_info(f"{self.script_prefix} Error in teardown: {e}", context=connection.id)

            # Clean up event handlers
            for _callback_id, (on_event_fired, model_name, event_name) in list(callbacks.items()):
                m = connection.get_model(model_name)
                if m:
                    m.off(event_name, on_event_fired)

            del self.contexts[connection.id]


def load_all_scripts():
    global LOADED_SCRIPTS
    LOADED_SCRIPTS = []

    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
    if not os.path.exists(scripts_dir):
        os.makedirs(scripts_dir, exist_ok=True)
        logger.log_info(f"Created scripts directory: {scripts_dir}")
        return

    # Load python scripts
    script_files = glob.glob(os.path.join(scripts_dir, "*.py"))
    for script_file in script_files:
        if os.path.basename(script_file) == "__init__.py":
            continue
        try:
            name = os.path.splitext(os.path.basename(script_file))[0]
            spec = importlib.util.spec_from_file_location(name, script_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                LOADED_SCRIPTS.append(module)
                logger.log_info(f"Loaded script: {script_file}")
        except Exception as e:
            logger.log_info(f"Error loading script {script_file}: {e}")

    # Load JS scripts
    script_files_js = glob.glob(os.path.join(scripts_dir, "*.js"))
    for script_file in script_files_js:
        try:
            js_script = JavaScriptScript(script_file)
            LOADED_SCRIPTS.append(js_script)
            logger.log_info(f"Loaded JS script: {script_file}")
        except Exception as e:
            logger.log_info(f"Error loading JS script {script_file}: {e}")


async def run_setup(connection):
    if getattr(connection, "script", None):
        try:
            js_script = JavaScriptScript(code=connection.script, name="dynamic_script.js")
            connection._compiled_script = js_script
            logger.log_info("Running setup in dynamic script", context=connection.id)
            if inspect.iscoroutinefunction(js_script.setup):
                await js_script.setup(connection)
            else:
                js_script.setup(connection)
        except Exception as e:
            logger.log_info(f"Error running setup in dynamic script: {e}", context=connection.id)

    for script in LOADED_SCRIPTS:
        if hasattr(script, "setup"):
            try:
                logger.log_info(f"Running setup in script: {script.__name__}")
                if inspect.iscoroutinefunction(script.setup):
                    await script.setup(connection)
                else:
                    script.setup(connection)
            except Exception as e:
                logger.log_info(f"Error running setup in script {script.__name__}: {e}")


async def run_teardown(connection):
    compiled_script = getattr(connection, "_compiled_script", None)
    if compiled_script and hasattr(compiled_script, "teardown"):
        try:
            logger.log_info("Running teardown in dynamic script", context=connection.id)
            if inspect.iscoroutinefunction(compiled_script.teardown):
                await compiled_script.teardown(connection)
            else:
                compiled_script.teardown(connection)
        except Exception as e:
            logger.log_info(f"Error running teardown in dynamic script: {e}", context=connection.id)

    for script in LOADED_SCRIPTS:
        if hasattr(script, "teardown"):
            try:
                logger.log_info(f"Running teardown in script: {script.__name__}")
                if inspect.iscoroutinefunction(script.teardown):
                    await script.teardown(connection)
                else:
                    script.teardown(connection)
            except Exception as e:
                logger.log_info(f"Error running teardown in script {script.__name__}: {e}")
