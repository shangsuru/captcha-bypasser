const express = require("express");
const app = express();
const path = require("path");
const captcha = require("nodejs-captcha");

app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));
app.use(express.static(__dirname + "/static"));

attempts = -1;

app.get("/", (req, res) => {
    attempts++;

    let result = captcha();
    let source = result.image;
    let value = result.value;
    res.render("index", { source, value, attempts });
});

app.listen(8000, "127.0.0.1", () => console.log("Listening on port 8000"));
