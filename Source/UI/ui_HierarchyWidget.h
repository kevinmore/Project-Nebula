/********************************************************************************
** Form generated from reading UI file 'HierarchyWidget.ui'
**
** Created by: Qt User Interface Compiler version 5.3.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_HIERARCHYWIDGET_H
#define UI_HIERARCHYWIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QTreeWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_HierarchyViewer
{
public:
    QVBoxLayout *verticalLayout;
    QTreeWidget *treeWidget;

    void setupUi(QWidget *HierarchyViewer)
    {
        if (HierarchyViewer->objectName().isEmpty())
            HierarchyViewer->setObjectName(QStringLiteral("HierarchyViewer"));
        HierarchyViewer->resize(387, 437);
        QFont font;
        font.setFamily(QString::fromUtf8("\345\276\256\350\275\257\351\233\205\351\273\221"));
        HierarchyViewer->setFont(font);
        verticalLayout = new QVBoxLayout(HierarchyViewer);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        treeWidget = new QTreeWidget(HierarchyViewer);
        treeWidget->setObjectName(QStringLiteral("treeWidget"));

        verticalLayout->addWidget(treeWidget);


        retranslateUi(HierarchyViewer);

        QMetaObject::connectSlotsByName(HierarchyViewer);
    } // setupUi

    void retranslateUi(QWidget *HierarchyViewer)
    {
        HierarchyViewer->setWindowTitle(QApplication::translate("HierarchyViewer", "Hierarchy Inspector", 0));
        QTreeWidgetItem *___qtreewidgetitem = treeWidget->headerItem();
        ___qtreewidgetitem->setText(0, QApplication::translate("HierarchyViewer", "1", 0));
    } // retranslateUi

};

namespace Ui {
    class HierarchyViewer: public Ui_HierarchyViewer {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_HIERARCHYWIDGET_H
