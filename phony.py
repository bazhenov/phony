import numpy.random as rnd
import math
import sys
import json

class Augmentator:

  formats = [
    ["#", "+#"],
    ["###", "(###)", '####', '(####)'],
    ["###-##-##", "### ## ##", "#######", '####-###', '### ###', '#####']
  ]

  def __init__(self):
    self.rules = []
    pass

  def format(self, phone, format=None, whitespaces=None):
    result = ""
    if format == None:
      format = self.choose_format(phone)
    if whitespaces == None:
      whitespaces = (rnd.randint(2) == 0)

    parts = []
    for i in range(3):
      format_part = format[i]
      parts.append(Augmentator.format_pattern(format_part, phone[i]))

    if whitespaces:
      result = " ".join(parts)
    else:
      result = "".join(parts)

    for rule in self.rules:
      result = rule(result)

    return result

  def setup_rules(self):
    self.rules.append(lambda p : p if rnd.random() < 0.5 else Augmentator.eight_rule(p))
    self.rules.append(lambda p : p if rnd.random() < 0.9 else Augmentator.random_punctuation_rule(p))
    self.rules.append(lambda p : p if rnd.random() < 0.9 else Augmentator.quasinumber_rule(p))

  def choose_format(self, phone):
    result = []
    for (phone_part, format_parts) in zip(phone, self.formats):
      phone_part_length = int(math.ceil(math.log10(phone_part)))
      valid_format_parts = [i for i in format_parts if i.count('#') == phone_part_length]
      if len(valid_format_parts) == 0:
        raise ValueError("No valid formats for phone " + str(phone))
      result.append(rnd.choice(valid_format_parts))
    return result

  def format_pattern(pattern, number):
    result = str(pattern)
    for chr in str(number):
      result = result.replace('#', chr, 1)
    return result

  def eight_rule(ph):
    """ Заменяет +7 на 8 """
    if ph.startswith("7"):
      return "8" + ph[1:]
    if ph.startswith("+7"):
      return "8" + ph[2:]
    return ph

  def random_punctuation_rule(ph, character=None, position=None, count=None):
    """ Вставляет символ пунктуации в произвольное место внутри строки """
    if character == None:
      character = rnd.choice(['/', '_', '-', '\\', '@', '#', '_', '.', ' '])
    if position == None:
      position = rnd.randint(1, len(ph))
    if count == None:
      count = rnd.randint(1, 10)
    return ph[0:position] + (character * count) + ph[position:]

  def quasinumber_rule(ph):
    """ Заменяет цифры на похожие по написанию буквы """
    mapping = {
      '4': ['Ч'],
      '0': ['O', 'О'],
      '1': ['l']
    }
    chars = list(ph)
    chars = [rnd.choice(mapping.get(i, [i])) for i in chars]
    return "".join(chars)

if __name__ == "__main__":
  a = Augmentator()
  a.setup_rules()
  placeholder = "<PHONE>"

  for line in sys.stdin:
    line = json.loads(line.strip())
    indexes = []
    while line.find(placeholder) >= 0:
      phone = (7, rnd.randint(900, 999 + 1), rnd.randint(1000000, 9999999 + 1))

      index = line.find(placeholder)
      phone_str = a.format(phone)
      line = line.replace(placeholder, phone_str, 1)
      indexes.append([index, index + len(phone_str)])
    print(json.dumps({'message': line, 'phone_indexes': indexes}, ensure_ascii=False))
