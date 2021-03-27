function checkCode(expected) {
    // Compare with code entered and display a message depending on if it is right or wrong
    let submitMsg = document.getElementById("submit-message");
    let input = document.getElementById("code").value;
    if (input == expected) {
        submitMsg.innerHTML = "Bypassed CAPTCHA successfully :)";
        submitMsg.classList.add("success");
    } else {
        submitMsg.innerHTML = "Failed at bypassing the CAPTCHA :(";
        submitMsg.classList.add("failure");
    }

    input.value = "";
}
