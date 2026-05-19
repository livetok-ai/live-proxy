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
    def __init__(self, file_path):
        self.file_path = file_path
        self.__name__ = os.path.basename(file_path)
        with open(file_path, encoding="utf-8") as f:
            self.code = f.read()
        self.contexts = {}  # connection.id -> (ctx, callbacks)

    def _setup_context(self, connection):
        if quickjs is None:
            raise RuntimeError("quickjs-ng package is not installed. Please run 'uv pip install quickjs-ng'")

        ctx = quickjs.Context()
        callbacks = {}  # callback_id -> (python_cb, model_name, event_name)

        def py_log(level, msg):
            if level == "error":
                logger.log_info(f"[{self.__name__}] [ERROR] {msg}", context=connection.id)
            else:
                logger.log_info(f"[{self.__name__}] {msg}", context=connection.id)

        ctx.add_callable("py_log", py_log)
        ctx.add_callable("py_connection_get_id", lambda: connection.id)
        ctx.add_callable("py_connection_has_model", lambda name: connection.get_model(name) is not None)

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

        ctx.add_callable("py_model_enable_input", py_model_enable_input)
        ctx.add_callable("py_model_disable_input", py_model_disable_input)
        ctx.add_callable("py_model_enable_output", py_model_enable_output)
        ctx.add_callable("py_model_disable_output", py_model_disable_output)

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
                        logger.log_info(f"[{self.__name__}] Error triggering JS callback: {e}", context=connection.id)

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
                    cb(data);
                } catch (e) {
                    py_log("error", "Error invoking JS callback: " + e + "\\n" + e.stack);
                }
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
        class ModelWrapper {
            constructor(name) {
                this.name = name;
            }
            enable_input() { py_model_enable_input(this.name); }
            disable_input() { py_model_disable_input(this.name); }
            enable_output() { py_model_enable_output(this.name); }
            disable_output() { py_model_disable_output(this.name); }
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
                    logger.log_info(f"[{self.__name__}] Error in teardown: {e}", context=connection.id)

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
