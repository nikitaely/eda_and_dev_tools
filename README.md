# Shoppers

**Online Shoppers Purchasing Intention Dataset**

Атрибут `Revenue` можно использовать в качестве метки класса.

`Administrative`, `Administrative Duration`, `Informational`, `Informational Duration`, `Product Related` и `Product Related Duration` представляют количество различных типов страниц, посещенных посетителем в этом сеансе, и общее время, проведенное в каждой из этих категорий страниц. Значения этих функций извлекаются из информации об URL-адресах страниц, посещенных пользователем, и обновляются в режиме реального времени, когда пользователь выполняет действие, например. переход с одной страницы на другую. 

`Bounce Rate`, `Exit Rate` и `Page Value` признаки представляют собой показатели, измеряемые «Google Analytics» для каждой страницы на сайте электронной коммерции. Значение признака `Bounce Rate` для веб-страницы относится к проценту посетителей, которые заходят на сайт с этой страницы, а затем покидают («bounce»), не инициируя никаких других запросов к серверу аналитики во время этого сеанса. Значение признака `Exit Rate` для конкретной веб-страницы рассчитывается как процент пользователей, для которых просмотр этой страницы был последним в сеансе. Признак `Page Value` представляет собой среднее значение для веб-страницы, которую пользователь посетил перед потверждением транзакции электронной торговли.

Функция `Special Day` указывает на близость времени посещения сайта к определенному особому дню (например, Дню матери, Дню святого Валентина), когда сеансы с большей вероятностью завершатся транзакцией. Значение этого атрибута определяется с учетом динамики электронной торговли, такой как продолжительность между датой заказа и датой доставки. Например, для дня Валентина это значение принимает ненулевое значение между 2 и 12 февраля, ноль до и после этой даты, если только она не близка к другому особому дню, и максимальное значение 1 приходится на 8 февраля.

Набор данных также включает операционную систему, браузер, регион, тип трафика, тип посетителя как вернувшегося или нового посетителя, логическое значение, указывающее, является ли дата посещения выходным, и месяц в году.

[Link](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) to the description of the original dataset.
