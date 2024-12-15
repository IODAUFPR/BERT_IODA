from flask import Flask, render_template, request, jsonify
from bert_model import bert_similarity  # Certifique-se de que este módulo está correto e importável

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        # Recebe os dados do formulário
        marca = request.form.get("marca", "").strip()
        colidencias = request.form.get("colidencias", "").strip()

        # Validação de entrada
        if not marca:
            return jsonify({"success": False, "error": "O campo 'Sua Marca' é obrigatório."})
        if not colidencias:
            return jsonify({"success": False, "error": "O campo 'Marcas Colidentes' é obrigatório."})

        # Divide as marcas colidentes por linha, removendo linhas vazias
        colidencias_list = [col.strip() for col in colidencias.split("\n") if col.strip()]

        if not colidencias_list:
            return jsonify({"success": False, "error": "Adicione ao menos uma marca colidente."})

        # Calcula a similaridade para cada marca colidente
        results = [
            {"colidencia": colidencia, "similarity": bert_similarity(marca, colidencia)}
            for colidencia in colidencias_list
        ]

        # Retorna os resultados como JSON
        return jsonify({"success": True, "results": results})
    
    except Exception as e:
        # Retorna o erro caso ocorra uma exceção
        return jsonify({"success": False, "error": f"Erro no servidor: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
