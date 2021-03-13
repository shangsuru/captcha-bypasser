const express = require("express");
const app = express();
const path = require("path");
const captcha = require("nodejs-captcha");

//app.engine("html", require("ejs").renderFile);
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));

app.get("/", (req, res) => {
    let result = captcha();
    let source = result.image;
    let value = result.value;
    res.render("index", { source, value });
});

app.get("/captcha", (req, res) => {
    let result = captcha();
    let source = result.image;
    res.end(
        `
    <!doctype html>
    <html>
        <head>
            <title>Test Captcha</title>
        </head>
        <body>
        <label>Test image</label>
        <img src="${source}" />
        <p>${result.value}</p>
        </body>
    </html>`
    );
});

app.listen(8000, "127.0.0.1");
