function checkCode(expected) {
    let submitMsg = document.getElementById("submit-message");
    let input = document.getElementById("code").value;
    if (input == expected) {
        submitMsg.innerHTML = "Bypassed CAPTCHA successfully :)";
        submitMsg.classList.add("success");
    } else {
        submitMsg.innerHTML = "Failed at bypassing the CAPTCHA :(";
        submitMsg.classList.add("failure");
    }
}
