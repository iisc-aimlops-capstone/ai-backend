from googletrans import Translator, LANGUAGES

async def translate_text(text, target):
    translator = Translator()

    # Await the coroutine
    translated = await translator.translate(text, dest=target)

    print(f"Original: \n{text}")
    print(f"Translated: \n{translated.text}")

    return {"response": translated.text}