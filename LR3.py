import sys
import os
import pickle
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QLabel, QMessageBox, QSpinBox, QFileDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPainter, QPen, QColor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback

model = None
vectorizer = None
label_encoder = None

CLASS_NAMES = {
    'ham': 'Не спам',
    'urgent': 'Срочное сообщение',
    'promotional': 'Промоакция',
    'phishing': 'Фишинг'
}

CLASS_COLORS = {
    'ham': QColor(76, 175, 80),
    'urgent': QColor(244, 67, 54),
    'promotional': QColor(255, 152, 0),
    'phishing': QColor(156, 39, 176)
}


class CircularProgressBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0
        self.max_value = 100
        self.label_text = ""
        self.is_selected = False
        self.color = QColor(33, 150, 243)
        self.setMinimumSize(150, 150)
        self.setMaximumSize(150, 150)
        
    def setValue(self, value):
        self.value = max(0, min(value, self.max_value))
        self.update()
        
    def setLabel(self, text):
        self.label_text = text
        self.update()
        
    def setColor(self, color):
        self.color = color
        self.update()
        
    def setSelected(self, selected):
        self.is_selected = selected
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        size = min(self.width(), self.height())
        rect_size = size - 20
        x = (self.width() - rect_size) // 2
        y = (self.height() - rect_size) // 2
        
        pen_width = 12 if self.is_selected else 10
        pen = QPen()
        pen.setWidth(pen_width)
        
        bg_color = QColor(240, 240, 240)
        pen.setColor(bg_color)
        painter.setPen(pen)
        painter.drawEllipse(x, y, rect_size, rect_size)
        
        if self.value > 0:
            angle = int(16 * 360 * self.value / self.max_value)
            pen.setColor(self.color)
            if self.is_selected:
                pen.setWidth(pen_width)
            painter.setPen(pen)
            painter.drawArc(x, y, rect_size, rect_size, 90 * 16, -angle)
        
        if self.is_selected:
            selected_pen = QPen(QColor(33, 150, 243), 3)
            painter.setPen(selected_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(x - 5, y - 5, rect_size + 10, rect_size + 10)
        
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        painter.setFont(font)
        
        percent_text = f"{self.value:.1f}%"
        text_pen = QPen(QColor(50, 50, 50))
        painter.setPen(text_pen)
        text_rect = self.rect()
        text_rect.setHeight(text_rect.height() // 2)
        painter.drawText(text_rect, Qt.AlignCenter, percent_text)
        
        font.setPointSize(9)
        font.setBold(False)
        painter.setFont(font)
        label_rect = self.rect()
        label_rect.setTop(label_rect.height() // 2)
        painter.drawText(label_rect, Qt.AlignCenter | Qt.TextWordWrap, self.label_text)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Классификатор текстовых сообщений')
        self.setGeometry(100, 100, 800, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        title = QLabel('Классификация текстовых сообщений')
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        params_layout = QHBoxLayout()
        params_label = QLabel('Параметры обучения:')
        params_label.setFont(QFont('Arial', 10, QFont.Bold))
        params_layout.addWidget(params_label)

        epochs_label = QLabel('Эпох:')
        params_layout.addWidget(epochs_label)
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setMinimum(1)
        self.epochs_spinbox.setMaximum(1000)
        self.epochs_spinbox.setValue(20)
        params_layout.addWidget(self.epochs_spinbox)

        batch_label = QLabel('Batch size:')
        params_layout.addWidget(batch_label)
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setMinimum(1)
        self.batch_spinbox.setMaximum(512)
        self.batch_spinbox.setValue(64)
        params_layout.addWidget(self.batch_spinbox)

        params_layout.addStretch()
        layout.addLayout(params_layout)

        buttons_layout = QHBoxLayout()
        train_btn = QPushButton('Обучить')
        train_btn.setMinimumHeight(40)
        train_btn.setFont(QFont('Arial', 12))
        train_btn.clicked.connect(
            lambda: train_model(self, self.train_result_text, self.epochs_spinbox.value(), self.batch_spinbox.value()))
        buttons_layout.addWidget(train_btn)

        save_btn = QPushButton('Сохранить модель')
        save_btn.setMinimumHeight(40)
        save_btn.setFont(QFont('Arial', 10))
        save_btn.clicked.connect(lambda: save_model(self))
        buttons_layout.addWidget(save_btn)

        load_btn = QPushButton('Загрузить модель')
        load_btn.setMinimumHeight(40)
        load_btn.setFont(QFont('Arial', 10))
        load_btn.clicked.connect(lambda: load_model_func(self))
        buttons_layout.addWidget(load_btn)
        layout.addLayout(buttons_layout)

        train_result_label = QLabel('Результаты обучения:')
        train_result_label.setFont(QFont('Arial', 10, QFont.Bold))
        layout.addWidget(train_result_label)

        self.train_result_text = QTextEdit()
        self.train_result_text.setMaximumHeight(150)
        self.train_result_text.setReadOnly(True)
        self.train_result_text.setPlaceholderText('Здесь будут отображаться результаты обучения')
        layout.addWidget(self.train_result_text)

        separator = QLabel('─' * 50)
        separator.setAlignment(Qt.AlignCenter)
        layout.addWidget(separator)

        input_label = QLabel('Введите текст письма для проверки:')
        input_label.setFont(QFont('Arial', 10, QFont.Bold))
        layout.addWidget(input_label)

        self.input_text = QTextEdit()
        self.input_text.setMinimumHeight(150)
        self.input_text.setPlaceholderText('Введите текст сообщения здесь')
        layout.addWidget(self.input_text)

        self.check_btn = QPushButton('Проверить')
        self.check_btn.setMinimumHeight(40)
        self.check_btn.setFont(QFont('Arial', 12))
        self.check_btn.setEnabled(False)
        self.check_btn.clicked.connect(
            lambda: check_text(self, self.input_text, self.result_widget, self.verdict_label))
        layout.addWidget(self.check_btn)

        result_label = QLabel('Результаты проверки:')
        result_label.setFont(QFont('Arial', 10, QFont.Bold))
        layout.addWidget(result_label)

        self.verdict_label = QLabel('')
        self.verdict_label.setAlignment(Qt.AlignCenter)
        self.verdict_label.setMinimumHeight(80)
        self.verdict_label.setWordWrap(True)
        layout.addWidget(self.verdict_label)

        self.result_widget = QWidget()
        result_layout = QVBoxLayout()
        result_layout.setContentsMargins(10, 10, 10, 10)
        self.result_widget.setLayout(result_layout)
        self.result_widget.setMinimumHeight(200)
        layout.addWidget(self.result_widget)

        self.statusBar().showMessage('Готово к работе. Нажмите "Обучить" для начала обучения.')


def save_model(window):
    global model, vectorizer, label_encoder
    
    if model is None or vectorizer is None or label_encoder is None:
        QMessageBox.warning(window, 'Предупреждение', 'Модель не обучена! Сначала обучите модель.')
        return
    
    filename, _ = QFileDialog.getSaveFileName(window, 'Сохранить модель', '', 'H5 Files (*.h5);;All Files (*)')
    if not filename:
        return
    
    try:
        model.save(filename)
        base_name = filename.rsplit('.', 1)[0]
        
        with open(f'{base_name}_vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        with open(f'{base_name}_label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        
        QMessageBox.information(window, 'Успех', f'Модель успешно сохранена:\n{filename}')
    except Exception as e:
        QMessageBox.critical(window, 'Ошибка', f'Ошибка при сохранении: {str(e)}')


def load_model_func(window):
    global model, vectorizer, label_encoder
    
    filename, _ = QFileDialog.getOpenFileName(window, 'Загрузить модель', '', 'H5 Files (*.h5);;All Files (*)')
    if not filename:
        return
    
    try:
        model = load_model(filename)
        base_name = filename.rsplit('.h5', 1)[0]
        
        with open(f'{base_name}_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open(f'{base_name}_label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        window.train_result_text.setText(f'Модель загружена из:\n{filename}')
        window.check_btn.setEnabled(True)
        QMessageBox.information(window, 'Успех', 'Модель успешно загружена!')
    except Exception as e:
        QMessageBox.critical(window, 'Ошибка', f'Ошибка при загрузке: {str(e)}')


def train_model(window, result_text, epochs, batch_size):
    global model, vectorizer, label_encoder
    
    if not os.path.exists('dataset.csv'):
        QMessageBox.critical(window, 'Ошибка', 'Файл dataset.csv не найден!')
        return
    
    result_text.clear()
    result_text.setText('Загрузка данных из dataset.csv.\n')
    QApplication.processEvents()
    
    df = pd.read_csv('dataset.csv')
    result_text.append(f'Загружено {len(df)} записей\n')
    QApplication.processEvents()
    
    result_text.append('Обработка данных.\n')
    QApplication.processEvents()
    
    texts = df['text'].values
    labels = df['label'].values
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    result_text.append('Векторизация текста...\n')
    QApplication.processEvents()
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), 
                                min_df=2, max_df=0.95)
    X = vectorizer.fit_transform(texts).toarray()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    result_text.append(f'Размер обучающей выборки: {len(X_train)}\n')
    result_text.append(f'Размер тестовой выборки: {len(X_test)}\n')
    result_text.append('Создание нейронной сети...\n')
    QApplication.processEvents()
    
    model = Sequential([
        Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer=SGD(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    result_text.append('Начало обучения\n')
    QApplication.processEvents()

    def on_epoch_end(epoch, logs):
        logs = logs or {}
        epoch_num = epoch + 1
        loss = logs.get('loss', 0)
        acc = logs.get('accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        
        msg = (f"Эпоха {epoch_num}/{epochs}: "
               f"loss={loss:.4f}, acc={acc:.4f}, "
               f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}\n")
        
        result_text.append(msg)
        QApplication.processEvents()

    progress_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=0,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[progress_callback, early_stopping],
        verbose=0
    )
    
    result_text.append('\nОценка модели...\n')
    QApplication.processEvents()
    
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    result = (f'\nОбучение завершено!\n\n'
             f'Обработано эпох: {len(history.history["loss"])}\n'
             f'Точность на обучающей выборке: {train_acc:.4f}\n'
             f'Точность на тестовой выборке: {test_acc:.4f}\n'
             f'Потери на тестовой выборке: {test_loss:.4f}')
    
    result_text.append(result)
    window.check_btn.setEnabled(True)
    QMessageBox.information(window, 'Успех', 'Модель успешно обучена!')

def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.deleteLater()
        elif child_layout is not None:
            clear_layout(child_layout)


def check_text(window, input_text, result_widget, verdict_label):
    global model, vectorizer, label_encoder
    
    if model is None or vectorizer is None or label_encoder is None:
        QMessageBox.warning(window, 'Предупреждение', 
                          'Модель не обучена! Необходимо сначала обучить модель.')
        return
    
    text = input_text.toPlainText().strip()
    if not text:
        QMessageBox.warning(window, 'Предупреждение', 'Введите текст для проверки!')
        return
    
    try:
        X = vectorizer.transform([text])
        predictions = model.predict(X, verbose=0)[0]
        class_names = label_encoder.classes_

        layout = result_widget.layout()
        clear_layout(layout)
        
        predicted_idx = np.argmax(predictions)
        predicted_class = class_names[predicted_idx]
        confidence = predictions[predicted_idx] * 100
        verdict_ru = CLASS_NAMES.get(predicted_class, predicted_class)
        
        verdict_label.setText(f'<h2 style="text-align: center; color: #1976D2;">Вердикт: {verdict_ru}</h2>'
                             f'<p style="text-align: center; color: #666;">Уверенность: {confidence:.2f}%</p>')
        
        circles_layout = QHBoxLayout()
        circles_layout.setSpacing(30)
        circles_layout.addStretch()
        
        sorted_indices = np.argsort(predictions)[::-1]
        for idx in sorted_indices:
            class_name = class_names[idx]
            probability = predictions[idx] * 100
            class_name_ru = CLASS_NAMES.get(class_name, class_name)
            
            circle = CircularProgressBar()
            circle.setValue(probability)
            circle.setLabel(class_name_ru)
            circle.setColor(CLASS_COLORS.get(class_name, QColor(33, 150, 243)))
            circle.setSelected(idx == predicted_idx)
            
            circles_layout.addWidget(circle)
        
        circles_layout.addStretch()
        result_widget.layout().addLayout(circles_layout)
        
    except Exception as e:
        QMessageBox.critical(window, 'Ошибка', f'Ошибка при классификации: {str(e)}')





def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
