from tensorflow import keras
import phony

if __name__ == "__main__":
  path = "./model"
  model = keras.models.load_model(path)

  phony.evaluate(model, "Привет это мой номер телефона: +7 914-705-7823 отдам за 2 миллиона", [])
  