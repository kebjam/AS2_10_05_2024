{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyP0kP7XgrEOgvOzRubezVJ2",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/kebjam/AS2_10_05_2024/blob/master/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8TUmzhE8c6I_",
        "outputId": "60bfdd9f-908c-45c8-ddba-683d665a67fa"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[?25l     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/981.5 kB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K     \u001b[91m━━━━━━━━━━━━━\u001b[0m\u001b[90m╺\u001b[0m\u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m327.7/981.5 kB\u001b[0m \u001b[31m9.7 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m\r\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m981.5/981.5 kB\u001b[0m \u001b[31m14.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h  Preparing metadata (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Preparing metadata (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m51.8/51.8 kB\u001b[0m \u001b[31m3.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m44.3/44.3 kB\u001b[0m \u001b[31m1.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m46.9/46.9 MB\u001b[0m \u001b[31m18.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m322.2/322.2 kB\u001b[0m \u001b[31m23.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m9.8/9.8 MB\u001b[0m \u001b[31m114.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m104.1/104.1 kB\u001b[0m \u001b[31m8.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m95.2/95.2 kB\u001b[0m \u001b[31m7.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m6.9/6.9 MB\u001b[0m \u001b[31m74.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m11.5/11.5 MB\u001b[0m \u001b[31m105.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m72.0/72.0 kB\u001b[0m \u001b[31m5.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m62.5/62.5 kB\u001b[0m \u001b[31m4.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m79.1/79.1 kB\u001b[0m \u001b[31m5.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h  Building wheel for langdetect (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Building wheel for rouge-score (setup.py) ... \u001b[?25l\u001b[?25hdone\n"
          ]
        }
      ],
      "source": [
        "!pip install transformers gradio streamlit langdetect rouge-score sacrebleu --quiet"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import streamlit as st\n",
        "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM\n",
        "from langdetect import detect\n",
        "\n",
        "# Configuration de la page Streamlit\n",
        "st.set_page_config(\n",
        "    page_title=\"Résumé Automatique (FR/EN)\",\n",
        "    layout=\"wide\"\n",
        ")\n",
        "\n",
        "st.title(\"Résumé Automatique (FR/EN)\")\n",
        "st.markdown(\"Collez un texte en français ou en anglais. Le modèle détectera la langue et générera un résumé dans la même langue.\")\n",
        "\n",
        "@st.cache_resource\n",
        "def load_models():\n",
        "    # Chargement des modèles\n",
        "    model_fr = AutoModelForSeq2SeqLM.from_pretrained(\"plguillou/t5-base-fr-sum-cnndm\")\n",
        "    tokenizer_fr = AutoTokenizer.from_pretrained(\"plguillou/t5-base-fr-sum-cnndm\")\n",
        "    model_en = AutoModelForSeq2SeqLM.from_pretrained(\"facebook/bart-large-cnn\")\n",
        "    tokenizer_en = AutoTokenizer.from_pretrained(\"facebook/bart-large-cnn\")\n",
        "    return model_fr, tokenizer_fr, model_en, tokenizer_en\n",
        "\n",
        "# Chargement des modèles avec mise en cache\n",
        "model_fr, tokenizer_fr, model_en, tokenizer_en = load_models()\n",
        "\n",
        "# Fonction de résumé\n",
        "def summarize(text):\n",
        "    if not text.strip():\n",
        "        return \"Veuillez entrer un texte à résumer.\"\n",
        "\n",
        "    try:\n",
        "        lang = detect(text)\n",
        "    except:\n",
        "        return \"Langue non détectée. Veuillez entrer un texte en français ou en anglais.\"\n",
        "\n",
        "    with st.spinner('Génération du résumé en cours...'):\n",
        "        if lang == 'fr':\n",
        "            input_ids = tokenizer_fr.encode(\"summarize: \" + text, return_tensors=\"pt\", max_length=1024, truncation=True)\n",
        "            output = model_fr.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)\n",
        "            summary = tokenizer_fr.decode(output[0], skip_special_tokens=True)\n",
        "            return f\"(Français détecté)\\n\\n{summary}\"\n",
        "        elif lang == 'en':\n",
        "            input_ids = tokenizer_en.encode(text, return_tensors=\"pt\", max_length=1024, truncation=True)\n",
        "            output = model_en.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)\n",
        "            summary = tokenizer_en.decode(output[0], skip_special_tokens=True)\n",
        "            return f\"(Anglais détecté)\\n\\n{summary}\"\n",
        "        else:\n",
        "            return \"Langue non prise en charge (seulement français ou anglais).\"\n",
        "\n",
        "# Interface utilisateur Streamlit\n",
        "col1, col2 = st.columns(2)\n",
        "\n",
        "with col1:\n",
        "    st.subheader(\"Texte à résumer\")\n",
        "    text_input = st.text_area(\"\", height=400)\n",
        "\n",
        "    if st.button(\"Générer le résumé\"):\n",
        "        with col2:\n",
        "            st.subheader(\"Résumé\")\n",
        "            result = summarize(text_input)\n",
        "            st.text_area(\"\", value=result, height=400, disabled=True)\n",
        "\n",
        "# Instructions pour exécuter sur Google Colab\n",
        "st.sidebar.title(\"Instructions pour Google Colab\")\n",
        "st.sidebar.markdown(\"\"\"\n",
        "Pour exécuter cette application sur Google Colab:\n",
        "\n",
        "1. Installez les dépendances nécessaires:\n",
        "```python\n",
        "!pip install transformers torch langdetect streamlit pyngrok\n",
        "```\n",
        "\n",
        "2. Sauvegardez ce code dans un fichier app.py:\n",
        "```python\n",
        "# Copiez le code de cette application\n",
        "```\n",
        "\n",
        "3. Exécutez l'application avec Ngrok:\n",
        "```python\n",
        "from pyngrok import ngrok\n",
        "!streamlit run app.py &>/dev/null&\n",
        "public_url = ngrok.connect(8501)\n",
        "print(f\"L'application est accessible à l'adresse: {public_url}\")\n",
        "```\n",
        "\"\"\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SqQoP-IcsMk8",
        "outputId": "decc8301-24b4-4dab-abf5-9fb9f9b9ed43"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2025-04-22 22:51:31.635 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.637 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.638 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.639 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.640 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.643 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.644 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.645 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.646 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.647 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.648 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.649 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.649 `label` got an empty value. This is discouraged for accessibility reasons and may be disallowed in the future by raising an exception. Please provide a non-empty label and hide it with label_visibility if needed.\n",
            "2025-04-22 22:51:31.650 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.651 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.652 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.653 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.654 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.655 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.656 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.656 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.657 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.658 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.659 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.660 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:51:31.661 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "DeltaGenerator(_root_container=1, _parent=DeltaGenerator())"
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    }
  ]
}