from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

def gauss_elimination(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    for i in range(n):
        max_row = i + np.argmax(np.abs(A[i:, i]))
        if A[max_row, i] == 0:
            raise ValueError("Matrix is singular or nearly singular")
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]
        for j in range(i + 1, n):
            ratio = A[j, i] / A[i, i]
            A[j] -= ratio * A[i]
            b[j] -= ratio * b[i]
    x = np.zeros_like(b)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.sum(A[i, i + 1:] * x[i + 1:])) / A[i, i]
    return x.tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            rows = int(request.form['rows'])
            columns = int(request.form['columns'])
            matrix = []
            b = []
            for i in range(rows):
                row = list(map(float, request.form.getlist(f'row{i}[]')))
                if len(row) != columns + 1:
                    raise ValueError(f"Expected {columns + 1} values in row {i}, but got {len(row)}")
                matrix.append(row[:columns])
                b.append(row[columns])
            solution = gauss_elimination(matrix, b)
            return render_template('index.html', solution=solution, rows=rows, columns=columns)
        except Exception as e:
            return render_template('index.html', error=str(e))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
