<!DOCTYPE html>
<html lang="cs">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dotazy k penzijku</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        .container {
            width: 90%;
            max-width: 800px;
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        h1 {
            text-align: center;
            color: #0099cc;
            margin-bottom: 20px;
        }

        .chat-history {
            max-height: 300px;
            overflow-y: auto;
            margin-bottom: 20px;
            padding-right: 10px;
            border-bottom: 1px solid #eee;
        }

        .message {
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
        }

        .user-message {
            background-color: #e1f5fe;
            color: #0099cc;
            text-align: left;
        }

        .assistant-message {
            background-color: #f0f4f8;
            color: #333;
            text-align: left;
            white-space: pre-wrap;  /* Aby se text formátoval správně */
        }

        textarea {
            width: 100%;
            padding: 10px;
            font-size: 16px;
            border: 1px solid #0099cc;
            border-radius: 4px;
            resize: none;
        }

        textarea:focus {
            outline: none;
            border-color: #006699;
        }

        button {
            width: 100%;
            padding: 12px;
            font-size: 16px;
            background-color: #0099cc;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 10px;
        }

        button:hover {
            background-color: #0077b3;
        }

        .error {
            color: red;
            text-align: center;
            margin-top: 10px;
        }

        /* Indikátor načítání */
        #loading {
            display: none;
            font-size: 18px;
            color: #007bff;
            text-align: center;
        }

        #spinner {
            display: none;
            margin: 0 auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Dotazy k penzijku</h1>
        <div class="chat-history" id="chatHistory"></div>
        <div class="input-group">
            <textarea id="query" placeholder="Co by Vás k penzijku zajímalo?"></textarea>
        </div>
        <button onclick="sendMessage()">Odeslat</button>
        <div id="loading">
            <p>Načítám odpověď, prosím čekejte...</p>
            <div id="spinner">🌀 Načítání...</div>
        </div>
        <div class="error" id="error"></div>
    </div>

    <script>
        async function sendMessage() {
            const query = document.getElementById('query').value;
            const errorBox = document.getElementById('error');
            const chatHistory = document.getElementById('chatHistory');
            const loadingIndicator = document.getElementById('loading');
            const spinner = document.getElementById('spinner');

            errorBox.textContent = "";

            if (!query.trim()) {
                errorBox.textContent = "Prosím, napište svůj dotaz.";
                return;
            }

            // Přidání dotazu do historie
            chatHistory.innerHTML += `<div class="message user-message"><strong>Vy:</strong> ${query}</div>`;
            document.getElementById('query').value = '';  // Vymazání textového pole

            // Zobrazíme indikátor načítání
            loadingIndicator.style.display = 'block';
            spinner.style.display = 'block';

            try {
                const response = await fetch("https://chatgpt-api-bawc.onrender.com", { // URL API upraveno na správnou adresu
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ query: query })
                });

                if (!response.ok) {
                    throw new Error(`Chyba: ${response.status} ${response.statusText}`);
                }

                const data = await response.json();
                if (data.answer) {
                    // Postupné zobrazování odpovědi po znacích
                    const answer = data.answer;
                    chatHistory.innerHTML += `<div class="message assistant-message"><strong>Asistentka:</strong> </div>`;
                    let charIndex = 0;
                    const messageDiv = chatHistory.lastElementChild;
                    
                    function addCharacter() {
                        if (charIndex < answer.length) {
                            messageDiv.innerHTML += answer.charAt(charIndex);
                            charIndex++;
                            setTimeout(addCharacter, 20); // Zpoždění 20ms mezi znaky
                        }
                    }

                    addCharacter();
                    chatHistory.scrollTop = chatHistory.scrollHeight;  // Scroll na konec chat history
                } else if (data.error) {
                    errorBox.textContent = data.error;
                } else {
                    errorBox.textContent = "Neznámá chyba při zpracování odpovědi.";
                }
            } catch (error) {
                errorBox.textContent = "Chyba při komunikaci se serverem: " + error.message;
            } finally {
                // Skrytí indikátoru načítání po obdržení odpovědi
                loadingIndicator.style.display = 'none';
                spinner.style.display = 'none';
            }
        }
    </script>
</body>
</html>
