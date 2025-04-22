{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOHvAPFxpBrtBGvi98i04MT",
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
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8TUmzhE8c6I_",
        "outputId": "06993001-21b9-43a7-9409-d53ec4b8129d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[?25l     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/981.5 kB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K     \u001b[91m━━━━━━━━━━━\u001b[0m\u001b[90m╺\u001b[0m\u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m286.7/981.5 kB\u001b[0m \u001b[31m9.1 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m\r\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m981.5/981.5 kB\u001b[0m \u001b[31m14.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h  Preparing metadata (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Preparing metadata (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m51.8/51.8 kB\u001b[0m \u001b[31m2.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m44.3/44.3 kB\u001b[0m \u001b[31m1.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m46.9/46.9 MB\u001b[0m \u001b[31m11.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m322.2/322.2 kB\u001b[0m \u001b[31m16.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m9.8/9.8 MB\u001b[0m \u001b[31m69.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m104.1/104.1 kB\u001b[0m \u001b[31m7.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m95.2/95.2 kB\u001b[0m \u001b[31m6.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m6.9/6.9 MB\u001b[0m \u001b[31m87.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m11.5/11.5 MB\u001b[0m \u001b[31m83.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m72.0/72.0 kB\u001b[0m \u001b[31m4.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m62.5/62.5 kB\u001b[0m \u001b[31m4.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m79.1/79.1 kB\u001b[0m \u001b[31m5.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
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
        "import torch\n",
        "\n",
        "# Configuration de la page\n",
        "st.set_page_config(\n",
        "    page_title=\"Résumé Automatique Bilingue\",\n",
        "    page_icon=\"📝\",\n",
        "    layout=\"wide\"\n",
        ")\n",
        "\n",
        "# Chargement optimisé des modèles\n",
        "@st.cache_resource(show_spinner=False)\n",
        "def load_models():\n",
        "    with st.spinner(\"Chargement des modèles (cela peut prendre 1-2 minutes)...\"):\n",
        "        try:\n",
        "            # Modèle français (plus léger)\n",
        "            model_fr = AutoModelForSeq2SeqLM.from_pretrained(\n",
        "                \"plguillou/t5-base-fr-sum-cnndm\",\n",
        "                device_map=\"auto\",\n",
        "                torch_dtype=torch.float16\n",
        "            )\n",
        "            tokenizer_fr = AutoTokenizer.from_pretrained(\"plguillou/t5-base-fr-sum-cnndm\")\n",
        "\n",
        "            # Modèle anglais\n",
        "            model_en = AutoModelForSeq2SeqLM.from_pretrained(\n",
        "                \"facebook/bart-large-cnn\",\n",
        "                device_map=\"auto\",\n",
        "                torch_dtype=torch.float16\n",
        "            )\n",
        "            tokenizer_en = AutoTokenizer.from_pretrained(\"facebook/bart-large-cnn\")\n",
        "\n",
        "            return model_fr, tokenizer_fr, model_en, tokenizer_en\n",
        "\n",
        "        except Exception as e:\n",
        "            st.error(f\"Erreur de chargement : {str(e)}\")\n",
        "            return None, None, None, None\n",
        "\n",
        "# Interface utilisateur\n",
        "st.title(\"📝 Résumé Automatique Français/Anglais\")\n",
        "st.markdown(\"\"\"\n",
        "<style>\n",
        ".stTextArea textarea {font-size: 16px !important;}\n",
        "</style>\n",
        "\"\"\", unsafe_allow_html=True)\n",
        "\n",
        "# Zone de texte\n",
        "text = st.text_area(\n",
        "    \"Collez votre texte ici (minimum 200 caractères) :\",\n",
        "    height=300,\n",
        "    placeholder=\"Le résumé automatique est une technique...\\n\\nAutomatic text summarization is...\"\n",
        ")\n",
        "\n",
        "# Bouton de génération\n",
        "if st.button(\"Générer le résumé\", type=\"primary\"):\n",
        "    if not text or len(text) < 200:\n",
        "        st.warning(\"Veuillez entrer un texte plus long (au moins 200 caractères)\")\n",
        "    else:\n",
        "        with st.spinner(\"Analyse en cours...\"):\n",
        "            try:\n",
        "                # Détection de langue\n",
        "                lang = detect(text)\n",
        "\n",
        "                if lang == 'fr':\n",
        "                    # Traitement français\n",
        "                    inputs = tokenizer_fr(\n",
        "                        \"résumer: \" + text,\n",
        "                        return_tensors=\"pt\",\n",
        "                        truncation=True,\n",
        "                        max_length=1024\n",
        "                    ).to(model_fr.device)\n",
        "\n",
        "                    outputs = model_fr.generate(\n",
        "                        **inputs,\n",
        "                        max_length=150,\n",
        "                        min_length=50,\n",
        "                        num_beams=4,\n",
        "                        length_penalty=2.0,\n",
        "                        early_stopping=True\n",
        "                    )\n",
        "                    summary = tokenizer_fr.decode(outputs[0], skip_special_tokens=True)\n",
        "                    st.success(\"**Résumé généré (Français) :**\")\n",
        "\n",
        "                elif lang == 'en':\n",
        "                    # Traitement anglais\n",
        "                    inputs = tokenizer_en(\n",
        "                        text,\n",
        "                        return_tensors=\"pt\",\n",
        "                        truncation=True,\n",
        "                        max_length=1024\n",
        "                    ).to(model_en.device)\n",
        "\n",
        "                    outputs = model_en.generate(\n",
        "                        **inputs,\n",
        "                        max_length=150,\n",
        "                        min_length=50,\n",
        "                        num_beams=4,\n",
        "                        length_penalty=2.0,\n",
        "                        early_stopping=True\n",
        "                    )\n",
        "                    summary = tokenizer_en.decode(outputs[0], skip_special_tokens=True)\n",
        "                    st.success(\"**Generated Summary (English) :**\")\n",
        "\n",
        "                else:\n",
        "                    st.error(\"Langue non supportée - Seul le français et l'anglais sont acceptés\")\n",
        "                    summary = \"\"\n",
        "\n",
        "                # Affichage du résultat\n",
        "                if summary:\n",
        "                    st.markdown(f\"\"\"\n",
        "                    <div style='\n",
        "                        padding: 15px;\n",
        "                        border-radius: 5px;\n",
        "                        background-color: #f0f2f6;\n",
        "                        border-left: 4px solid #4e79a7;\n",
        "                        margin-top: 10px;\n",
        "                    '>\n",
        "                    {summary}\n",
        "                    </div>\n",
        "                    \"\"\", unsafe_allow_html=True)\n",
        "                    st.caption(f\"Longueur : {len(summary.split())} mots\")\n",
        "\n",
        "            except Exception as e:\n",
        "                st.error(f\"Erreur lors de la génération : {str(e)}\")\n",
        "\n",
        "# Pied de page\n",
        "st.markdown(\"---\")\n",
        "st.caption(\"Application développée avec 🤗 Transformers et Streamlit\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zwNR2uHQdESk",
        "outputId": "02236cba-c332-4a3f-9e5e-c21a06ba4fb0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2025-04-22 22:19:12.235 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:19:12.238 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:19:12.349 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py [ARGUMENTS]\n",
            "2025-04-22 22:19:12.349 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:19:12.350 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:19:12.351 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:19:12.352 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:19:12.352 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:19:12.353 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:19:12.354 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:19:12.354 Session state does not function when running a script without `streamlit run`\n",
            "2025-04-22 22:19:12.355 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:19:12.356 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:19:12.357 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:19:12.357 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:19:12.358 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:19:12.359 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:19:12.359 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:19:12.360 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:19:12.361 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:19:12.362 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-04-22 22:19:12.362 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "DeltaGenerator()"
            ]
          },
          "metadata": {},
          "execution_count": 2
        }
      ]
    }
  ]
}