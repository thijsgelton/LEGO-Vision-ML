# Localization with SVM
The Support Vector Machine style Localization pulls from the <a href="LEGO Vision/Detection/SVM"> SVM Detection </a> folder.

Except from using a config that points to a different dataset, the only different thing are the Selective Search parameters. This is because with the localization problem the form of the objects is different. This results in different parameters which tend to prefer larger objects more.