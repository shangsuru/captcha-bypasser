const captcha = require("nodejs-captcha");
fs = require("fs");

// Run this to create a sample dataset of CAPTCHA images

const numExamples = 10000;

for (let i = 0; i < numExamples; i++) {
    let result = captcha();
    let source = result.image;
    let value = result.value;

    var base64Data = source.replace(/^data:image\/jpeg;base64,/, "");

    let dir = "dataset/images";
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir);
    }

    fs.writeFile(`${dir}/${value}.jpeg`, base64Data, "base64", function (err) {});
}
