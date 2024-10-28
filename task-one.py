from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus


def main():
    model = LpProblem(name="optimization-of-drink-production", sense=LpMaximize)

    lemonade = LpVariable(name="lemonade", lowBound=0, cat="Integer")
    fruit_juice = LpVariable(name="fruit_juice", lowBound=0, cat="Integer")

    model += (2 * lemonade + 1 * fruit_juice <= 100, "water_constraint")
    model += (1 * lemonade <= 50, "sugar_constraint")
    model += (1 * lemonade <= 30, "lemon_juice_constraint")
    model += (2 * fruit_juice <= 40, "fruit_puree_constraint")

    model += lpSum([lemonade, fruit_juice])
    status = model.solve()

    return lemonade.value(), fruit_juice.value(), LpStatus[status]


if __name__ == "__main__":
    lemonade_qty, fruit_juice_qty, status = main()

    print(f"Статус роботи: {status}")
    print(f"Лимонаду потрібно виробити: {lemonade_qty}")
    print(f"Фруктового соку потрібно виробити: {fruit_juice_qty}")
    print(f"Загальна кількість продуктів: {lemonade_qty + fruit_juice_qty}")
