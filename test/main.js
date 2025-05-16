const { app, BrowserWindow } = require("electron");

function createWindow() {
    function loadScript(fileName) {
        win.webContents.executeJavaScript(`
                const script = document.querySelector('script');
                if (script) script.src = '${fileName}';
            `);
    }

    const win = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
        },
    });

    win.loadFile("index.html");
    win.webContents.once("did-finish-load", () => {
        if (process.argv.includes("bench")) {
            loadScript("bench.js");
        } else if (process.argv.includes("layout")) {
            loadScript("test_layout.js");
        } else {
            loadScript("test.js");
        }
    });

    win.webContents.openDevTools();
}

app.whenReady().then(() => {
    createWindow();

    app.on("activate", () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on("window-all-closed", () => {
    if (process.platform !== "darwin") {
        app.quit();
    }
});
