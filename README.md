# Описание решения

В main.py представлен код для обучения модели, в inference.py для получения предсказания, в main.ipynb предварительный анализ данных и запуск пердсказания.

Файл data/submision.csv содержит ответ на задание. Дополнительные столбцы 'prob_mean', 'prob_mean_per_pos', 'prob_max' содержат соотвественно информацию о:

1. средней вероятности позитивного класса
2. средней вероятности среди значений, выше трешхолда 0.5
3. максимальной вероятности

(Подробнее написано в методе)
# Метод

В качестве модели была выбрана [rubert-base](https://huggingface.co/DeepPavlov/rubert-base-cased). 
Из-за проблем с переобучением была также попробована [маленькая](https://huggingface.co/cointegrated/rubert-tiny2) модель и классические методы машинного обучения.
Первая все еще оказалась склонна к переобучению, а вторые показывали не такое хорошее качество.

Так как датасет несблансирован, то функция потерь (кроссэнтропия) взвешивается пропорционально классам.

Так как в тестовом наборе данных тексты имеют значительно большую длину, то для предсказания текст новости разбивается на предложения (тут есть сложности из-за наличия
сокращений внутри предложений), которые и поступают на вход модели. Таким образом для одной новости мы получаем несколько пердсказаний.
Так как очевидно, что интересующая нас информация может быть соредоточена только в одном предложении, то усреденение всех вероятностей (prob_mean)
является не очень информативной оценкой. Эту проблему можно решить взятием среднего среди значений выше трешхолда 0.5 (prob_mean_per_pos), однако для тогда 
все предложения негативного класса становятся неотличимы. Частично эту проблему решает взятие максимума (prob_max) среди предсказанных вероятностей, однако
такая оценка является слишком точечной. 

Как показал анализ результирующего файла data/submission.csv, сортировка именно по prob_max является наиболее информативной, однако для полноты картины рекомендуется
учитывать во внимание все три столбца.