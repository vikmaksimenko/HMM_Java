HMM_Java
========

Realization of Hidden Markov Model for gesture recognition on Java. Based on GRT

Главный класс - Main пакета hmmMain содержит методы для распознавания фигур. Для этого выполняется подгрузка сериализованных объектов KMeansQuantizer (квантайзер, приводящий тип данных к соответствующему Марковской модели) и HMM (обертка для Марковских моделей, содержащая метод predict() для распознавания).

Алгоритм работы: 

1. Загрузка квантайзера и модели
2. Загрузка данных
3. Квантование данных
4. Распознавание
5. Вывод результата

Для упрощения вывода был написан метод printLabel(), принимающий номер класса и выводящий номер фигуры. Также соответствие "номер-название" можно найти в 3 строке (InfoText) файла HMMTrainingDataACC1.txt

Данный код написан с использованием GRT (<Nicholas Gillian, Media Lab, MIT>, )

The HMM_Java is available under a MIT license.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
