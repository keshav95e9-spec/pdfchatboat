console.log("PDF RAG Chatbot Loaded");

const uploadInput = document.querySelector('input[type="file"]');

uploadInput.addEventListener("change", () => {

    if(uploadInput.files.length > 0){

        const fileName = uploadInput.files[0].name;

        alert("Selected File: " + fileName);
    }
});