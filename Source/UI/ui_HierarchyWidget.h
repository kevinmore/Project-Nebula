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
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTreeWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_HierarchyViewer
{
public:
    QVBoxLayout *verticalLayout;
    QTreeWidget *treeWidget;
    QTabWidget *tabWidget;
    QWidget *TransformTab;
    QVBoxLayout *verticalLayout_5;
    QHBoxLayout *horizontalLayout_4;
    QVBoxLayout *verticalLayout_2;
    QLabel *label;
    QHBoxLayout *horizontalLayout;
    QLabel *label_4;
    QDoubleSpinBox *doubleSpinBox_PositionX;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label_5;
    QDoubleSpinBox *doubleSpinBox_PositionY;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_6;
    QDoubleSpinBox *doubleSpinBox_PositionZ;
    QVBoxLayout *verticalLayout_3;
    QLabel *label_2;
    QHBoxLayout *horizontalLayout_6;
    QLabel *label_7;
    QDoubleSpinBox *doubleSpinBox_RotationX;
    QHBoxLayout *horizontalLayout_7;
    QLabel *label_8;
    QDoubleSpinBox *doubleSpinBox_RotationY;
    QHBoxLayout *horizontalLayout_8;
    QLabel *label_9;
    QDoubleSpinBox *doubleSpinBox_RotationZ;
    QVBoxLayout *verticalLayout_4;
    QLabel *label_3;
    QHBoxLayout *horizontalLayout_10;
    QLabel *label_10;
    QDoubleSpinBox *doubleSpinBox_ScaleX;
    QHBoxLayout *horizontalLayout_11;
    QLabel *label_11;
    QDoubleSpinBox *doubleSpinBox_ScaleY;
    QHBoxLayout *horizontalLayout_12;
    QLabel *label_12;
    QDoubleSpinBox *doubleSpinBox_ScaleZ;
    QSpacerItem *verticalSpacer;
    QPushButton *pushButton_Reset;
    QWidget *RenderTab;

    void setupUi(QWidget *HierarchyViewer)
    {
        if (HierarchyViewer->objectName().isEmpty())
            HierarchyViewer->setObjectName(QStringLiteral("HierarchyViewer"));
        HierarchyViewer->setEnabled(true);
        HierarchyViewer->resize(220, 826);
        QSizePolicy sizePolicy(QSizePolicy::Minimum, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(HierarchyViewer->sizePolicy().hasHeightForWidth());
        HierarchyViewer->setSizePolicy(sizePolicy);
        HierarchyViewer->setMinimumSize(QSize(220, 0));
        QFont font;
        font.setFamily(QString::fromUtf8("\345\276\256\350\275\257\351\233\205\351\273\221"));
        HierarchyViewer->setFont(font);
        verticalLayout = new QVBoxLayout(HierarchyViewer);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        treeWidget = new QTreeWidget(HierarchyViewer);
        treeWidget->setObjectName(QStringLiteral("treeWidget"));

        verticalLayout->addWidget(treeWidget);

        tabWidget = new QTabWidget(HierarchyViewer);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        QSizePolicy sizePolicy1(QSizePolicy::Minimum, QSizePolicy::Expanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(tabWidget->sizePolicy().hasHeightForWidth());
        tabWidget->setSizePolicy(sizePolicy1);
        tabWidget->setMinimumSize(QSize(220, 0));
        TransformTab = new QWidget();
        TransformTab->setObjectName(QStringLiteral("TransformTab"));
        verticalLayout_5 = new QVBoxLayout(TransformTab);
        verticalLayout_5->setObjectName(QStringLiteral("verticalLayout_5"));
        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QStringLiteral("horizontalLayout_4"));
        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        label = new QLabel(TransformTab);
        label->setObjectName(QStringLiteral("label"));
        label->setAlignment(Qt::AlignCenter);

        verticalLayout_2->addWidget(label);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        label_4 = new QLabel(TransformTab);
        label_4->setObjectName(QStringLiteral("label_4"));

        horizontalLayout->addWidget(label_4);

        doubleSpinBox_PositionX = new QDoubleSpinBox(TransformTab);
        doubleSpinBox_PositionX->setObjectName(QStringLiteral("doubleSpinBox_PositionX"));
        QSizePolicy sizePolicy2(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(doubleSpinBox_PositionX->sizePolicy().hasHeightForWidth());
        doubleSpinBox_PositionX->setSizePolicy(sizePolicy2);
        doubleSpinBox_PositionX->setMinimumSize(QSize(0, 0));
        doubleSpinBox_PositionX->setMinimum(-100000);
        doubleSpinBox_PositionX->setMaximum(100000);
        doubleSpinBox_PositionX->setSingleStep(50);

        horizontalLayout->addWidget(doubleSpinBox_PositionX);


        verticalLayout_2->addLayout(horizontalLayout);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        label_5 = new QLabel(TransformTab);
        label_5->setObjectName(QStringLiteral("label_5"));

        horizontalLayout_2->addWidget(label_5);

        doubleSpinBox_PositionY = new QDoubleSpinBox(TransformTab);
        doubleSpinBox_PositionY->setObjectName(QStringLiteral("doubleSpinBox_PositionY"));
        sizePolicy2.setHeightForWidth(doubleSpinBox_PositionY->sizePolicy().hasHeightForWidth());
        doubleSpinBox_PositionY->setSizePolicy(sizePolicy2);
        doubleSpinBox_PositionY->setMinimum(-100000);
        doubleSpinBox_PositionY->setMaximum(100000);
        doubleSpinBox_PositionY->setSingleStep(50);

        horizontalLayout_2->addWidget(doubleSpinBox_PositionY);


        verticalLayout_2->addLayout(horizontalLayout_2);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        label_6 = new QLabel(TransformTab);
        label_6->setObjectName(QStringLiteral("label_6"));

        horizontalLayout_3->addWidget(label_6);

        doubleSpinBox_PositionZ = new QDoubleSpinBox(TransformTab);
        doubleSpinBox_PositionZ->setObjectName(QStringLiteral("doubleSpinBox_PositionZ"));
        sizePolicy2.setHeightForWidth(doubleSpinBox_PositionZ->sizePolicy().hasHeightForWidth());
        doubleSpinBox_PositionZ->setSizePolicy(sizePolicy2);
        doubleSpinBox_PositionZ->setMinimum(-100000);
        doubleSpinBox_PositionZ->setSingleStep(50);
        doubleSpinBox_PositionZ->setValue(0);

        horizontalLayout_3->addWidget(doubleSpinBox_PositionZ);


        verticalLayout_2->addLayout(horizontalLayout_3);


        horizontalLayout_4->addLayout(verticalLayout_2);

        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        label_2 = new QLabel(TransformTab);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setAlignment(Qt::AlignCenter);

        verticalLayout_3->addWidget(label_2);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setObjectName(QStringLiteral("horizontalLayout_6"));
        label_7 = new QLabel(TransformTab);
        label_7->setObjectName(QStringLiteral("label_7"));

        horizontalLayout_6->addWidget(label_7);

        doubleSpinBox_RotationX = new QDoubleSpinBox(TransformTab);
        doubleSpinBox_RotationX->setObjectName(QStringLiteral("doubleSpinBox_RotationX"));
        sizePolicy2.setHeightForWidth(doubleSpinBox_RotationX->sizePolicy().hasHeightForWidth());
        doubleSpinBox_RotationX->setSizePolicy(sizePolicy2);
        doubleSpinBox_RotationX->setMinimum(-360);
        doubleSpinBox_RotationX->setMaximum(360);
        doubleSpinBox_RotationX->setSingleStep(10);

        horizontalLayout_6->addWidget(doubleSpinBox_RotationX);


        verticalLayout_3->addLayout(horizontalLayout_6);

        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setObjectName(QStringLiteral("horizontalLayout_7"));
        label_8 = new QLabel(TransformTab);
        label_8->setObjectName(QStringLiteral("label_8"));

        horizontalLayout_7->addWidget(label_8);

        doubleSpinBox_RotationY = new QDoubleSpinBox(TransformTab);
        doubleSpinBox_RotationY->setObjectName(QStringLiteral("doubleSpinBox_RotationY"));
        sizePolicy2.setHeightForWidth(doubleSpinBox_RotationY->sizePolicy().hasHeightForWidth());
        doubleSpinBox_RotationY->setSizePolicy(sizePolicy2);
        doubleSpinBox_RotationY->setMinimum(-360);
        doubleSpinBox_RotationY->setMaximum(360);
        doubleSpinBox_RotationY->setSingleStep(10);

        horizontalLayout_7->addWidget(doubleSpinBox_RotationY);


        verticalLayout_3->addLayout(horizontalLayout_7);

        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setObjectName(QStringLiteral("horizontalLayout_8"));
        label_9 = new QLabel(TransformTab);
        label_9->setObjectName(QStringLiteral("label_9"));

        horizontalLayout_8->addWidget(label_9);

        doubleSpinBox_RotationZ = new QDoubleSpinBox(TransformTab);
        doubleSpinBox_RotationZ->setObjectName(QStringLiteral("doubleSpinBox_RotationZ"));
        sizePolicy2.setHeightForWidth(doubleSpinBox_RotationZ->sizePolicy().hasHeightForWidth());
        doubleSpinBox_RotationZ->setSizePolicy(sizePolicy2);
        doubleSpinBox_RotationZ->setMinimum(-360);
        doubleSpinBox_RotationZ->setMaximum(360);
        doubleSpinBox_RotationZ->setSingleStep(10);

        horizontalLayout_8->addWidget(doubleSpinBox_RotationZ);


        verticalLayout_3->addLayout(horizontalLayout_8);


        horizontalLayout_4->addLayout(verticalLayout_3);

        verticalLayout_4 = new QVBoxLayout();
        verticalLayout_4->setObjectName(QStringLiteral("verticalLayout_4"));
        label_3 = new QLabel(TransformTab);
        label_3->setObjectName(QStringLiteral("label_3"));
        QSizePolicy sizePolicy3(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(label_3->sizePolicy().hasHeightForWidth());
        label_3->setSizePolicy(sizePolicy3);
        label_3->setMinimumSize(QSize(0, 0));
        label_3->setAlignment(Qt::AlignCenter);

        verticalLayout_4->addWidget(label_3);

        horizontalLayout_10 = new QHBoxLayout();
        horizontalLayout_10->setObjectName(QStringLiteral("horizontalLayout_10"));
        label_10 = new QLabel(TransformTab);
        label_10->setObjectName(QStringLiteral("label_10"));

        horizontalLayout_10->addWidget(label_10);

        doubleSpinBox_ScaleX = new QDoubleSpinBox(TransformTab);
        doubleSpinBox_ScaleX->setObjectName(QStringLiteral("doubleSpinBox_ScaleX"));
        sizePolicy2.setHeightForWidth(doubleSpinBox_ScaleX->sizePolicy().hasHeightForWidth());
        doubleSpinBox_ScaleX->setSizePolicy(sizePolicy2);
        doubleSpinBox_ScaleX->setMinimum(0);
        doubleSpinBox_ScaleX->setMaximum(100000);
        doubleSpinBox_ScaleX->setSingleStep(0.1);
        doubleSpinBox_ScaleX->setValue(1);

        horizontalLayout_10->addWidget(doubleSpinBox_ScaleX);


        verticalLayout_4->addLayout(horizontalLayout_10);

        horizontalLayout_11 = new QHBoxLayout();
        horizontalLayout_11->setObjectName(QStringLiteral("horizontalLayout_11"));
        label_11 = new QLabel(TransformTab);
        label_11->setObjectName(QStringLiteral("label_11"));

        horizontalLayout_11->addWidget(label_11);

        doubleSpinBox_ScaleY = new QDoubleSpinBox(TransformTab);
        doubleSpinBox_ScaleY->setObjectName(QStringLiteral("doubleSpinBox_ScaleY"));
        sizePolicy2.setHeightForWidth(doubleSpinBox_ScaleY->sizePolicy().hasHeightForWidth());
        doubleSpinBox_ScaleY->setSizePolicy(sizePolicy2);
        doubleSpinBox_ScaleY->setMinimum(0);
        doubleSpinBox_ScaleY->setMaximum(100000);
        doubleSpinBox_ScaleY->setSingleStep(0.1);
        doubleSpinBox_ScaleY->setValue(1);

        horizontalLayout_11->addWidget(doubleSpinBox_ScaleY);


        verticalLayout_4->addLayout(horizontalLayout_11);

        horizontalLayout_12 = new QHBoxLayout();
        horizontalLayout_12->setObjectName(QStringLiteral("horizontalLayout_12"));
        label_12 = new QLabel(TransformTab);
        label_12->setObjectName(QStringLiteral("label_12"));

        horizontalLayout_12->addWidget(label_12);

        doubleSpinBox_ScaleZ = new QDoubleSpinBox(TransformTab);
        doubleSpinBox_ScaleZ->setObjectName(QStringLiteral("doubleSpinBox_ScaleZ"));
        sizePolicy2.setHeightForWidth(doubleSpinBox_ScaleZ->sizePolicy().hasHeightForWidth());
        doubleSpinBox_ScaleZ->setSizePolicy(sizePolicy2);
        doubleSpinBox_ScaleZ->setMinimum(0);
        doubleSpinBox_ScaleZ->setMaximum(100000);
        doubleSpinBox_ScaleZ->setSingleStep(0.1);
        doubleSpinBox_ScaleZ->setValue(1);

        horizontalLayout_12->addWidget(doubleSpinBox_ScaleZ);


        verticalLayout_4->addLayout(horizontalLayout_12);


        horizontalLayout_4->addLayout(verticalLayout_4);


        verticalLayout_5->addLayout(horizontalLayout_4);

        verticalSpacer = new QSpacerItem(20, 199, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_5->addItem(verticalSpacer);

        pushButton_Reset = new QPushButton(TransformTab);
        pushButton_Reset->setObjectName(QStringLiteral("pushButton_Reset"));

        verticalLayout_5->addWidget(pushButton_Reset);

        tabWidget->addTab(TransformTab, QString());
        label->raise();
        label_2->raise();
        label_3->raise();
        doubleSpinBox_PositionX->raise();
        label_4->raise();
        doubleSpinBox_PositionY->raise();
        pushButton_Reset->raise();
        RenderTab = new QWidget();
        RenderTab->setObjectName(QStringLiteral("RenderTab"));
        tabWidget->addTab(RenderTab, QString());

        verticalLayout->addWidget(tabWidget);


        retranslateUi(HierarchyViewer);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(HierarchyViewer);
    } // setupUi

    void retranslateUi(QWidget *HierarchyViewer)
    {
        HierarchyViewer->setWindowTitle(QApplication::translate("HierarchyViewer", "Hierarchy Inspector", 0));
        QTreeWidgetItem *___qtreewidgetitem = treeWidget->headerItem();
        ___qtreewidgetitem->setText(0, QApplication::translate("HierarchyViewer", "Game Objects", 0));
        label->setText(QApplication::translate("HierarchyViewer", "Position", 0));
        label_4->setText(QApplication::translate("HierarchyViewer", "X", 0));
        label_5->setText(QApplication::translate("HierarchyViewer", "Y", 0));
        label_6->setText(QApplication::translate("HierarchyViewer", "Z", 0));
        label_2->setText(QApplication::translate("HierarchyViewer", "Rotation", 0));
        label_7->setText(QApplication::translate("HierarchyViewer", "X", 0));
        label_8->setText(QApplication::translate("HierarchyViewer", "Y", 0));
        label_9->setText(QApplication::translate("HierarchyViewer", "Z", 0));
        label_3->setText(QApplication::translate("HierarchyViewer", "Scale", 0));
        label_10->setText(QApplication::translate("HierarchyViewer", "X", 0));
        label_11->setText(QApplication::translate("HierarchyViewer", "Y", 0));
        label_12->setText(QApplication::translate("HierarchyViewer", "Z", 0));
        pushButton_Reset->setText(QApplication::translate("HierarchyViewer", "Reset", 0));
        tabWidget->setTabText(tabWidget->indexOf(TransformTab), QApplication::translate("HierarchyViewer", "Transformation", 0));
        tabWidget->setTabText(tabWidget->indexOf(RenderTab), QApplication::translate("HierarchyViewer", "Rendering", 0));
    } // retranslateUi

};

namespace Ui {
    class HierarchyViewer: public Ui_HierarchyViewer {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_HIERARCHYWIDGET_H
