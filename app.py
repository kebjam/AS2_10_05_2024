{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNJPwGNxD/ax+DxuN5IF/dO",
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
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8TUmzhE8c6I_",
        "outputId": "5554c939-32fd-4859-e95f-ac470ba23f3b"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[?25l     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/981.5 kB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K     \u001b[91m━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m\u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m317.4/981.5 kB\u001b[0m \u001b[31m9.7 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m\r\u001b[2K     \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m \u001b[32m972.8/981.5 kB\u001b[0m \u001b[31m18.0 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m\r\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m981.5/981.5 kB\u001b[0m \u001b[31m12.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h  Preparing metadata (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Preparing metadata (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m51.8/51.8 kB\u001b[0m \u001b[31m3.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h\u001b[33m  WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))': /packages/f9/b6/a447b5e4ec71e13871be01ba81f5dfc9d0af7e473da256ff46bc0e24026f/tomlkit-0.13.2-py3-none-any.whl.metadata\u001b[0m\u001b[33m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m46.9/46.9 MB\u001b[0m \u001b[31m15.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m322.2/322.2 kB\u001b[0m \u001b[31m16.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m104.1/104.1 kB\u001b[0m \u001b[31m6.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m95.2/95.2 kB\u001b[0m \u001b[31m6.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m11.5/11.5 MB\u001b[0m \u001b[31m76.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m72.0/72.0 kB\u001b[0m \u001b[31m3.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m62.5/62.5 kB\u001b[0m \u001b[31m3.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h  Building wheel for langdetect (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Building wheel for rouge-score (setup.py) ... \u001b[?25l\u001b[?25hdone\n"
          ]
        }
      ],
      "source": [
        "!pip install transformers gradio langdetect rouge-score sacrebleu --quiet"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM\n",
        "import gradio as gr\n",
        "from langdetect import detect\n",
        "\n",
        "# Chargement des modèles\n",
        "model_fr = AutoModelForSeq2SeqLM.from_pretrained(\"plguillou/t5-base-fr-sum-cnndm\")\n",
        "tokenizer_fr = AutoTokenizer.from_pretrained(\"plguillou/t5-base-fr-sum-cnndm\")\n",
        "\n",
        "model_en = AutoModelForSeq2SeqLM.from_pretrained(\"facebook/bart-large-cnn\")\n",
        "tokenizer_en = AutoTokenizer.from_pretrained(\"facebook/bart-large-cnn\")\n",
        "\n",
        "# Fonction de résumé\n",
        "def summarize(text):\n",
        "    try:\n",
        "        lang = detect(text)\n",
        "    except:\n",
        "        return \"Langue non détectée. Veuillez entrer un texte en français ou en anglais.\"\n",
        "\n",
        "    if lang == 'fr':\n",
        "        input_ids = tokenizer_fr.encode(\"summarize: \" + text, return_tensors=\"pt\", max_length=1024, truncation=True)\n",
        "        output = model_fr.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)\n",
        "        summary = tokenizer_fr.decode(output[0], skip_special_tokens=True)\n",
        "        return f\"(Français détecté)\\n\\n{summary}\"\n",
        "    elif lang == 'en':\n",
        "        input_ids = tokenizer_en.encode(text, return_tensors=\"pt\", max_length=1024, truncation=True)\n",
        "        output = model_en.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)\n",
        "        summary = tokenizer_en.decode(output[0], skip_special_tokens=True)\n",
        "        return f\"(Anglais détecté)\\n\\n{summary}\"\n",
        "    else:\n",
        "        return \"Langue non prise en charge (seulement français ou anglais).\"\n",
        "\n",
        "# Interface Gradio\n",
        "interface = gr.Interface(\n",
        "    fn=summarize,\n",
        "    inputs=gr.Textbox(lines=12, label=\"Texte à résumer\"),\n",
        "    outputs=gr.Textbox(label=\"Résumé\"),\n",
        "    title=\"Résumé Automatique (FR/EN)\",\n",
        "    description=\"Collez un texte en français ou en anglais. Le modèle détectera la langue et générera un résumé dans la même langue.\"\n",
        ")\n",
        "\n",
        "interface.launch(debug=True)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 647
        },
        "id": "zwNR2uHQdESk",
        "outputId": "c2f9342d-1e0a-4650-a35f-bb9358b91cf0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "It looks like you are running Gradio on a hosted a Jupyter notebook. For the Gradio app to work, sharing must be enabled. Automatically setting `share=True` (you can turn this off by setting `share=False` in `launch()` explicitly).\n",
            "\n",
            "Colab notebook detected. This cell will run indefinitely so that you can see errors and logs. To turn off, set debug=False in launch().\n",
            "* Running on public URL: https://369866d35c37eba5c3.gradio.live\n",
            "\n",
            "This share link expires in 1 week. For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "<div><iframe src=\"https://369866d35c37eba5c3.gradio.live\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
            ]
          },
          "metadata": {}
        }
      ]
    }
  ]
}