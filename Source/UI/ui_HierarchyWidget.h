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
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
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
    QVBoxLayout *verticalLayout_6;
    QScrollArea *scrollArea;
    QWidget *scrollAreaWidgetContents;
    QWidget *ParticleTab;
    QVBoxLayout *verticalLayout_7;
    QScrollArea *scrollArea_2;
    QWidget *scrollAreaWidgetContents_2;
    QVBoxLayout *verticalLayout_8;
    QGridLayout *gridLayout;
    QDoubleSpinBox *doubleSpinBox_EmitRate;
    QLabel *label_13;
    QLabel *label_32;
    QDoubleSpinBox *doubleSpinBox_Mass;
    QLabel *label_18;
    QLabel *label_14;
    QDoubleSpinBox *doubleSpinBox_Size;
    QDoubleSpinBox *doubleSpinBox_GravityFactor;
    QLabel *label_17;
    QLabel *label_15;
    QHBoxLayout *horizontalLayout_15;
    QSlider *horizontalSlider_EmitAmount;
    QSpinBox *spinBox_EmitAmount;
    QDoubleSpinBox *doubleSpinBox_MinLife;
    QDoubleSpinBox *doubleSpinBox_MaxLife;
    QLabel *label_19;
    QLabel *label_34;
    QHBoxLayout *horizontalLayout_5;
    QCheckBox *checkBox_RandomColor;
    QGraphicsView *graphicsView_ColorPicker;
    QHBoxLayout *horizontalLayout_9;
    QLabel *label_36;
    QGraphicsView *graphicsView_TexturePicker;
    QHBoxLayout *horizontalLayout_19;
    QVBoxLayout *verticalLayout_9;
    QLabel *label_20;
    QHBoxLayout *horizontalLayout_20;
    QLabel *label_21;
    QDoubleSpinBox *doubleSpinBox_ForceX;
    QHBoxLayout *horizontalLayout_21;
    QLabel *label_22;
    QDoubleSpinBox *doubleSpinBox_ForceY;
    QHBoxLayout *horizontalLayout_22;
    QLabel *label_23;
    QDoubleSpinBox *doubleSpinBox_ForceZ;
    QVBoxLayout *verticalLayout_10;
    QLabel *label_24;
    QHBoxLayout *horizontalLayout_23;
    QLabel *label_25;
    QDoubleSpinBox *doubleSpinBox_MinVelocityX;
    QHBoxLayout *horizontalLayout_24;
    QLabel *label_26;
    QDoubleSpinBox *doubleSpinBox_MinVelocityY;
    QHBoxLayout *horizontalLayout_25;
    QLabel *label_27;
    QDoubleSpinBox *doubleSpinBox_MinVelocityZ;
    QVBoxLayout *verticalLayout_11;
    QLabel *label_28;
    QHBoxLayout *horizontalLayout_26;
    QLabel *label_29;
    QDoubleSpinBox *doubleSpinBox_MaxVelocityX;
    QHBoxLayout *horizontalLayout_27;
    QLabel *label_30;
    QDoubleSpinBox *doubleSpinBox_MaxVelocityY;
    QHBoxLayout *horizontalLayout_28;
    QLabel *label_31;
    QDoubleSpinBox *doubleSpinBox_MaxVelocityZ;

    void setupUi(QWidget *HierarchyViewer)
    {
        if (HierarchyViewer->objectName().isEmpty())
            HierarchyViewer->setObjectName(QStringLiteral("HierarchyViewer"));
        HierarchyViewer->setEnabled(true);
        HierarchyViewer->resize(332, 863);
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
        tabWidget->setEnabled(true);
        sizePolicy.setHeightForWidth(tabWidget->sizePolicy().hasHeightForWidth());
        tabWidget->setSizePolicy(sizePolicy);
        tabWidget->setMinimumSize(QSize(220, 0));
        tabWidget->setTabShape(QTabWidget::Rounded);
        tabWidget->setElideMode(Qt::ElideNone);
        tabWidget->setDocumentMode(false);
        tabWidget->setTabsClosable(false);
        tabWidget->setMovable(true);
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
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(doubleSpinBox_PositionX->sizePolicy().hasHeightForWidth());
        doubleSpinBox_PositionX->setSizePolicy(sizePolicy1);
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
        sizePolicy1.setHeightForWidth(doubleSpinBox_PositionY->sizePolicy().hasHeightForWidth());
        doubleSpinBox_PositionY->setSizePolicy(sizePolicy1);
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
        sizePolicy1.setHeightForWidth(doubleSpinBox_PositionZ->sizePolicy().hasHeightForWidth());
        doubleSpinBox_PositionZ->setSizePolicy(sizePolicy1);
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
        sizePolicy1.setHeightForWidth(doubleSpinBox_RotationX->sizePolicy().hasHeightForWidth());
        doubleSpinBox_RotationX->setSizePolicy(sizePolicy1);
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
        sizePolicy1.setHeightForWidth(doubleSpinBox_RotationY->sizePolicy().hasHeightForWidth());
        doubleSpinBox_RotationY->setSizePolicy(sizePolicy1);
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
        sizePolicy1.setHeightForWidth(doubleSpinBox_RotationZ->sizePolicy().hasHeightForWidth());
        doubleSpinBox_RotationZ->setSizePolicy(sizePolicy1);
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
        QSizePolicy sizePolicy2(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(label_3->sizePolicy().hasHeightForWidth());
        label_3->setSizePolicy(sizePolicy2);
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
        sizePolicy1.setHeightForWidth(doubleSpinBox_ScaleX->sizePolicy().hasHeightForWidth());
        doubleSpinBox_ScaleX->setSizePolicy(sizePolicy1);
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
        sizePolicy1.setHeightForWidth(doubleSpinBox_ScaleY->sizePolicy().hasHeightForWidth());
        doubleSpinBox_ScaleY->setSizePolicy(sizePolicy1);
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
        sizePolicy1.setHeightForWidth(doubleSpinBox_ScaleZ->sizePolicy().hasHeightForWidth());
        doubleSpinBox_ScaleZ->setSizePolicy(sizePolicy1);
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
        pushButton_Reset->raise();
        RenderTab = new QWidget();
        RenderTab->setObjectName(QStringLiteral("RenderTab"));
        verticalLayout_6 = new QVBoxLayout(RenderTab);
        verticalLayout_6->setObjectName(QStringLiteral("verticalLayout_6"));
        scrollArea = new QScrollArea(RenderTab);
        scrollArea->setObjectName(QStringLiteral("scrollArea"));
        scrollArea->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QStringLiteral("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 288, 402));
        scrollArea->setWidget(scrollAreaWidgetContents);

        verticalLayout_6->addWidget(scrollArea);

        tabWidget->addTab(RenderTab, QString());
        ParticleTab = new QWidget();
        ParticleTab->setObjectName(QStringLiteral("ParticleTab"));
        verticalLayout_7 = new QVBoxLayout(ParticleTab);
        verticalLayout_7->setObjectName(QStringLiteral("verticalLayout_7"));
        scrollArea_2 = new QScrollArea(ParticleTab);
        scrollArea_2->setObjectName(QStringLiteral("scrollArea_2"));
        scrollArea_2->setWidgetResizable(true);
        scrollAreaWidgetContents_2 = new QWidget();
        scrollAreaWidgetContents_2->setObjectName(QStringLiteral("scrollAreaWidgetContents_2"));
        scrollAreaWidgetContents_2->setGeometry(QRect(0, -108, 323, 632));
        verticalLayout_8 = new QVBoxLayout(scrollAreaWidgetContents_2);
        verticalLayout_8->setObjectName(QStringLiteral("verticalLayout_8"));
        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        doubleSpinBox_EmitRate = new QDoubleSpinBox(scrollAreaWidgetContents_2);
        doubleSpinBox_EmitRate->setObjectName(QStringLiteral("doubleSpinBox_EmitRate"));
        doubleSpinBox_EmitRate->setMaximum(1e+06);
        doubleSpinBox_EmitRate->setSingleStep(0.01);
        doubleSpinBox_EmitRate->setValue(0);

        gridLayout->addWidget(doubleSpinBox_EmitRate, 3, 1, 1, 1);

        label_13 = new QLabel(scrollAreaWidgetContents_2);
        label_13->setObjectName(QStringLiteral("label_13"));
        label_13->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        gridLayout->addWidget(label_13, 0, 0, 1, 1);

        label_32 = new QLabel(scrollAreaWidgetContents_2);
        label_32->setObjectName(QStringLiteral("label_32"));
        label_32->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        gridLayout->addWidget(label_32, 1, 0, 1, 1);

        doubleSpinBox_Mass = new QDoubleSpinBox(scrollAreaWidgetContents_2);
        doubleSpinBox_Mass->setObjectName(QStringLiteral("doubleSpinBox_Mass"));
        doubleSpinBox_Mass->setMaximum(1e+06);
        doubleSpinBox_Mass->setValue(0);

        gridLayout->addWidget(doubleSpinBox_Mass, 0, 1, 1, 1);

        label_18 = new QLabel(scrollAreaWidgetContents_2);
        label_18->setObjectName(QStringLiteral("label_18"));
        label_18->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        gridLayout->addWidget(label_18, 5, 0, 1, 1);

        label_14 = new QLabel(scrollAreaWidgetContents_2);
        label_14->setObjectName(QStringLiteral("label_14"));
        label_14->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        gridLayout->addWidget(label_14, 2, 0, 1, 1);

        doubleSpinBox_Size = new QDoubleSpinBox(scrollAreaWidgetContents_2);
        doubleSpinBox_Size->setObjectName(QStringLiteral("doubleSpinBox_Size"));
        doubleSpinBox_Size->setMaximum(1e+06);
        doubleSpinBox_Size->setValue(0);

        gridLayout->addWidget(doubleSpinBox_Size, 2, 1, 1, 1);

        doubleSpinBox_GravityFactor = new QDoubleSpinBox(scrollAreaWidgetContents_2);
        doubleSpinBox_GravityFactor->setObjectName(QStringLiteral("doubleSpinBox_GravityFactor"));
        doubleSpinBox_GravityFactor->setMinimum(-1e+06);
        doubleSpinBox_GravityFactor->setMaximum(1e+06);
        doubleSpinBox_GravityFactor->setValue(1);

        gridLayout->addWidget(doubleSpinBox_GravityFactor, 1, 1, 1, 1);

        label_17 = new QLabel(scrollAreaWidgetContents_2);
        label_17->setObjectName(QStringLiteral("label_17"));
        label_17->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        gridLayout->addWidget(label_17, 4, 0, 1, 1);

        label_15 = new QLabel(scrollAreaWidgetContents_2);
        label_15->setObjectName(QStringLiteral("label_15"));
        label_15->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        gridLayout->addWidget(label_15, 3, 0, 1, 1);

        horizontalLayout_15 = new QHBoxLayout();
        horizontalLayout_15->setObjectName(QStringLiteral("horizontalLayout_15"));
        horizontalSlider_EmitAmount = new QSlider(scrollAreaWidgetContents_2);
        horizontalSlider_EmitAmount->setObjectName(QStringLiteral("horizontalSlider_EmitAmount"));
        horizontalSlider_EmitAmount->setMaximum(40);
        horizontalSlider_EmitAmount->setValue(0);
        horizontalSlider_EmitAmount->setOrientation(Qt::Horizontal);
        horizontalSlider_EmitAmount->setInvertedAppearance(false);
        horizontalSlider_EmitAmount->setInvertedControls(false);
        horizontalSlider_EmitAmount->setTickPosition(QSlider::NoTicks);

        horizontalLayout_15->addWidget(horizontalSlider_EmitAmount);

        spinBox_EmitAmount = new QSpinBox(scrollAreaWidgetContents_2);
        spinBox_EmitAmount->setObjectName(QStringLiteral("spinBox_EmitAmount"));
        spinBox_EmitAmount->setMaximum(40);
        spinBox_EmitAmount->setValue(0);

        horizontalLayout_15->addWidget(spinBox_EmitAmount);


        gridLayout->addLayout(horizontalLayout_15, 4, 1, 1, 1);

        doubleSpinBox_MinLife = new QDoubleSpinBox(scrollAreaWidgetContents_2);
        doubleSpinBox_MinLife->setObjectName(QStringLiteral("doubleSpinBox_MinLife"));
        doubleSpinBox_MinLife->setMaximum(1e+06);
        doubleSpinBox_MinLife->setSingleStep(1);
        doubleSpinBox_MinLife->setValue(0);

        gridLayout->addWidget(doubleSpinBox_MinLife, 5, 1, 1, 1);

        doubleSpinBox_MaxLife = new QDoubleSpinBox(scrollAreaWidgetContents_2);
        doubleSpinBox_MaxLife->setObjectName(QStringLiteral("doubleSpinBox_MaxLife"));
        doubleSpinBox_MaxLife->setMaximum(1e+06);
        doubleSpinBox_MaxLife->setSingleStep(1);
        doubleSpinBox_MaxLife->setValue(0);

        gridLayout->addWidget(doubleSpinBox_MaxLife, 6, 1, 1, 1);

        label_19 = new QLabel(scrollAreaWidgetContents_2);
        label_19->setObjectName(QStringLiteral("label_19"));
        label_19->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        gridLayout->addWidget(label_19, 6, 0, 1, 1);

        label_34 = new QLabel(scrollAreaWidgetContents_2);
        label_34->setObjectName(QStringLiteral("label_34"));
        sizePolicy2.setHeightForWidth(label_34->sizePolicy().hasHeightForWidth());
        label_34->setSizePolicy(sizePolicy2);
        label_34->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        gridLayout->addWidget(label_34, 7, 0, 1, 1);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName(QStringLiteral("horizontalLayout_5"));
        checkBox_RandomColor = new QCheckBox(scrollAreaWidgetContents_2);
        checkBox_RandomColor->setObjectName(QStringLiteral("checkBox_RandomColor"));

        horizontalLayout_5->addWidget(checkBox_RandomColor);

        graphicsView_ColorPicker = new QGraphicsView(scrollAreaWidgetContents_2);
        graphicsView_ColorPicker->setObjectName(QStringLiteral("graphicsView_ColorPicker"));
        QSizePolicy sizePolicy3(QSizePolicy::Minimum, QSizePolicy::Minimum);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(graphicsView_ColorPicker->sizePolicy().hasHeightForWidth());
        graphicsView_ColorPicker->setSizePolicy(sizePolicy3);
        graphicsView_ColorPicker->setMinimumSize(QSize(20, 20));

        horizontalLayout_5->addWidget(graphicsView_ColorPicker);


        gridLayout->addLayout(horizontalLayout_5, 7, 1, 1, 1);


        verticalLayout_8->addLayout(gridLayout);

        horizontalLayout_9 = new QHBoxLayout();
        horizontalLayout_9->setObjectName(QStringLiteral("horizontalLayout_9"));
        label_36 = new QLabel(scrollAreaWidgetContents_2);
        label_36->setObjectName(QStringLiteral("label_36"));
        sizePolicy2.setHeightForWidth(label_36->sizePolicy().hasHeightForWidth());
        label_36->setSizePolicy(sizePolicy2);
        label_36->setAlignment(Qt::AlignCenter);

        horizontalLayout_9->addWidget(label_36);

        graphicsView_TexturePicker = new QGraphicsView(scrollAreaWidgetContents_2);
        graphicsView_TexturePicker->setObjectName(QStringLiteral("graphicsView_TexturePicker"));
        sizePolicy3.setHeightForWidth(graphicsView_TexturePicker->sizePolicy().hasHeightForWidth());
        graphicsView_TexturePicker->setSizePolicy(sizePolicy3);
        graphicsView_TexturePicker->setMinimumSize(QSize(256, 256));

        horizontalLayout_9->addWidget(graphicsView_TexturePicker);


        verticalLayout_8->addLayout(horizontalLayout_9);

        horizontalLayout_19 = new QHBoxLayout();
        horizontalLayout_19->setObjectName(QStringLiteral("horizontalLayout_19"));
        verticalLayout_9 = new QVBoxLayout();
        verticalLayout_9->setObjectName(QStringLiteral("verticalLayout_9"));
        label_20 = new QLabel(scrollAreaWidgetContents_2);
        label_20->setObjectName(QStringLiteral("label_20"));
        label_20->setAlignment(Qt::AlignCenter);

        verticalLayout_9->addWidget(label_20);

        horizontalLayout_20 = new QHBoxLayout();
        horizontalLayout_20->setObjectName(QStringLiteral("horizontalLayout_20"));
        label_21 = new QLabel(scrollAreaWidgetContents_2);
        label_21->setObjectName(QStringLiteral("label_21"));

        horizontalLayout_20->addWidget(label_21);

        doubleSpinBox_ForceX = new QDoubleSpinBox(scrollAreaWidgetContents_2);
        doubleSpinBox_ForceX->setObjectName(QStringLiteral("doubleSpinBox_ForceX"));
        sizePolicy1.setHeightForWidth(doubleSpinBox_ForceX->sizePolicy().hasHeightForWidth());
        doubleSpinBox_ForceX->setSizePolicy(sizePolicy1);
        doubleSpinBox_ForceX->setMinimumSize(QSize(0, 0));
        doubleSpinBox_ForceX->setMinimum(-100000);
        doubleSpinBox_ForceX->setMaximum(100000);
        doubleSpinBox_ForceX->setSingleStep(10);

        horizontalLayout_20->addWidget(doubleSpinBox_ForceX);


        verticalLayout_9->addLayout(horizontalLayout_20);

        horizontalLayout_21 = new QHBoxLayout();
        horizontalLayout_21->setObjectName(QStringLiteral("horizontalLayout_21"));
        label_22 = new QLabel(scrollAreaWidgetContents_2);
        label_22->setObjectName(QStringLiteral("label_22"));

        horizontalLayout_21->addWidget(label_22);

        doubleSpinBox_ForceY = new QDoubleSpinBox(scrollAreaWidgetContents_2);
        doubleSpinBox_ForceY->setObjectName(QStringLiteral("doubleSpinBox_ForceY"));
        sizePolicy1.setHeightForWidth(doubleSpinBox_ForceY->sizePolicy().hasHeightForWidth());
        doubleSpinBox_ForceY->setSizePolicy(sizePolicy1);
        doubleSpinBox_ForceY->setMinimum(-100000);
        doubleSpinBox_ForceY->setMaximum(100000);
        doubleSpinBox_ForceY->setSingleStep(10);
        doubleSpinBox_ForceY->setValue(0);

        horizontalLayout_21->addWidget(doubleSpinBox_ForceY);


        verticalLayout_9->addLayout(horizontalLayout_21);

        horizontalLayout_22 = new QHBoxLayout();
        horizontalLayout_22->setObjectName(QStringLiteral("horizontalLayout_22"));
        label_23 = new QLabel(scrollAreaWidgetContents_2);
        label_23->setObjectName(QStringLiteral("label_23"));

        horizontalLayout_22->addWidget(label_23);

        doubleSpinBox_ForceZ = new QDoubleSpinBox(scrollAreaWidgetContents_2);
        doubleSpinBox_ForceZ->setObjectName(QStringLiteral("doubleSpinBox_ForceZ"));
        sizePolicy1.setHeightForWidth(doubleSpinBox_ForceZ->sizePolicy().hasHeightForWidth());
        doubleSpinBox_ForceZ->setSizePolicy(sizePolicy1);
        doubleSpinBox_ForceZ->setMinimum(-100000);
        doubleSpinBox_ForceZ->setMaximum(100000);
        doubleSpinBox_ForceZ->setSingleStep(10);
        doubleSpinBox_ForceZ->setValue(0);

        horizontalLayout_22->addWidget(doubleSpinBox_ForceZ);


        verticalLayout_9->addLayout(horizontalLayout_22);


        horizontalLayout_19->addLayout(verticalLayout_9);

        verticalLayout_10 = new QVBoxLayout();
        verticalLayout_10->setObjectName(QStringLiteral("verticalLayout_10"));
        label_24 = new QLabel(scrollAreaWidgetContents_2);
        label_24->setObjectName(QStringLiteral("label_24"));
        label_24->setAlignment(Qt::AlignCenter);

        verticalLayout_10->addWidget(label_24);

        horizontalLayout_23 = new QHBoxLayout();
        horizontalLayout_23->setObjectName(QStringLiteral("horizontalLayout_23"));
        label_25 = new QLabel(scrollAreaWidgetContents_2);
        label_25->setObjectName(QStringLiteral("label_25"));

        horizontalLayout_23->addWidget(label_25);

        doubleSpinBox_MinVelocityX = new QDoubleSpinBox(scrollAreaWidgetContents_2);
        doubleSpinBox_MinVelocityX->setObjectName(QStringLiteral("doubleSpinBox_MinVelocityX"));
        sizePolicy1.setHeightForWidth(doubleSpinBox_MinVelocityX->sizePolicy().hasHeightForWidth());
        doubleSpinBox_MinVelocityX->setSizePolicy(sizePolicy1);
        doubleSpinBox_MinVelocityX->setMinimum(-100000);
        doubleSpinBox_MinVelocityX->setMaximum(100000);
        doubleSpinBox_MinVelocityX->setSingleStep(1);
        doubleSpinBox_MinVelocityX->setValue(0);

        horizontalLayout_23->addWidget(doubleSpinBox_MinVelocityX);


        verticalLayout_10->addLayout(horizontalLayout_23);

        horizontalLayout_24 = new QHBoxLayout();
        horizontalLayout_24->setObjectName(QStringLiteral("horizontalLayout_24"));
        label_26 = new QLabel(scrollAreaWidgetContents_2);
        label_26->setObjectName(QStringLiteral("label_26"));

        horizontalLayout_24->addWidget(label_26);

        doubleSpinBox_MinVelocityY = new QDoubleSpinBox(scrollAreaWidgetContents_2);
        doubleSpinBox_MinVelocityY->setObjectName(QStringLiteral("doubleSpinBox_MinVelocityY"));
        sizePolicy1.setHeightForWidth(doubleSpinBox_MinVelocityY->sizePolicy().hasHeightForWidth());
        doubleSpinBox_MinVelocityY->setSizePolicy(sizePolicy1);
        doubleSpinBox_MinVelocityY->setMinimum(-100000);
        doubleSpinBox_MinVelocityY->setMaximum(100000);
        doubleSpinBox_MinVelocityY->setSingleStep(1);
        doubleSpinBox_MinVelocityY->setValue(0);

        horizontalLayout_24->addWidget(doubleSpinBox_MinVelocityY);


        verticalLayout_10->addLayout(horizontalLayout_24);

        horizontalLayout_25 = new QHBoxLayout();
        horizontalLayout_25->setObjectName(QStringLiteral("horizontalLayout_25"));
        label_27 = new QLabel(scrollAreaWidgetContents_2);
        label_27->setObjectName(QStringLiteral("label_27"));

        horizontalLayout_25->addWidget(label_27);

        doubleSpinBox_MinVelocityZ = new QDoubleSpinBox(scrollAreaWidgetContents_2);
        doubleSpinBox_MinVelocityZ->setObjectName(QStringLiteral("doubleSpinBox_MinVelocityZ"));
        sizePolicy1.setHeightForWidth(doubleSpinBox_MinVelocityZ->sizePolicy().hasHeightForWidth());
        doubleSpinBox_MinVelocityZ->setSizePolicy(sizePolicy1);
        doubleSpinBox_MinVelocityZ->setMinimum(-100000);
        doubleSpinBox_MinVelocityZ->setMaximum(100000);
        doubleSpinBox_MinVelocityZ->setSingleStep(1);
        doubleSpinBox_MinVelocityZ->setValue(0);

        horizontalLayout_25->addWidget(doubleSpinBox_MinVelocityZ);


        verticalLayout_10->addLayout(horizontalLayout_25);


        horizontalLayout_19->addLayout(verticalLayout_10);

        verticalLayout_11 = new QVBoxLayout();
        verticalLayout_11->setObjectName(QStringLiteral("verticalLayout_11"));
        label_28 = new QLabel(scrollAreaWidgetContents_2);
        label_28->setObjectName(QStringLiteral("label_28"));
        sizePolicy2.setHeightForWidth(label_28->sizePolicy().hasHeightForWidth());
        label_28->setSizePolicy(sizePolicy2);
        label_28->setMinimumSize(QSize(0, 0));
        label_28->setAlignment(Qt::AlignCenter);

        verticalLayout_11->addWidget(label_28);

        horizontalLayout_26 = new QHBoxLayout();
        horizontalLayout_26->setObjectName(QStringLiteral("horizontalLayout_26"));
        label_29 = new QLabel(scrollAreaWidgetContents_2);
        label_29->setObjectName(QStringLiteral("label_29"));

        horizontalLayout_26->addWidget(label_29);

        doubleSpinBox_MaxVelocityX = new QDoubleSpinBox(scrollAreaWidgetContents_2);
        doubleSpinBox_MaxVelocityX->setObjectName(QStringLiteral("doubleSpinBox_MaxVelocityX"));
        sizePolicy1.setHeightForWidth(doubleSpinBox_MaxVelocityX->sizePolicy().hasHeightForWidth());
        doubleSpinBox_MaxVelocityX->setSizePolicy(sizePolicy1);
        doubleSpinBox_MaxVelocityX->setMinimum(-100000);
        doubleSpinBox_MaxVelocityX->setMaximum(100000);
        doubleSpinBox_MaxVelocityX->setSingleStep(1);
        doubleSpinBox_MaxVelocityX->setValue(0);

        horizontalLayout_26->addWidget(doubleSpinBox_MaxVelocityX);


        verticalLayout_11->addLayout(horizontalLayout_26);

        horizontalLayout_27 = new QHBoxLayout();
        horizontalLayout_27->setObjectName(QStringLiteral("horizontalLayout_27"));
        label_30 = new QLabel(scrollAreaWidgetContents_2);
        label_30->setObjectName(QStringLiteral("label_30"));

        horizontalLayout_27->addWidget(label_30);

        doubleSpinBox_MaxVelocityY = new QDoubleSpinBox(scrollAreaWidgetContents_2);
        doubleSpinBox_MaxVelocityY->setObjectName(QStringLiteral("doubleSpinBox_MaxVelocityY"));
        sizePolicy1.setHeightForWidth(doubleSpinBox_MaxVelocityY->sizePolicy().hasHeightForWidth());
        doubleSpinBox_MaxVelocityY->setSizePolicy(sizePolicy1);
        doubleSpinBox_MaxVelocityY->setMinimum(-100000);
        doubleSpinBox_MaxVelocityY->setMaximum(100000);
        doubleSpinBox_MaxVelocityY->setSingleStep(1);
        doubleSpinBox_MaxVelocityY->setValue(0);

        horizontalLayout_27->addWidget(doubleSpinBox_MaxVelocityY);


        verticalLayout_11->addLayout(horizontalLayout_27);

        horizontalLayout_28 = new QHBoxLayout();
        horizontalLayout_28->setObjectName(QStringLiteral("horizontalLayout_28"));
        label_31 = new QLabel(scrollAreaWidgetContents_2);
        label_31->setObjectName(QStringLiteral("label_31"));

        horizontalLayout_28->addWidget(label_31);

        doubleSpinBox_MaxVelocityZ = new QDoubleSpinBox(scrollAreaWidgetContents_2);
        doubleSpinBox_MaxVelocityZ->setObjectName(QStringLiteral("doubleSpinBox_MaxVelocityZ"));
        sizePolicy1.setHeightForWidth(doubleSpinBox_MaxVelocityZ->sizePolicy().hasHeightForWidth());
        doubleSpinBox_MaxVelocityZ->setSizePolicy(sizePolicy1);
        doubleSpinBox_MaxVelocityZ->setMinimum(-100000);
        doubleSpinBox_MaxVelocityZ->setMaximum(100000);
        doubleSpinBox_MaxVelocityZ->setSingleStep(1);
        doubleSpinBox_MaxVelocityZ->setValue(0);

        horizontalLayout_28->addWidget(doubleSpinBox_MaxVelocityZ);


        verticalLayout_11->addLayout(horizontalLayout_28);


        horizontalLayout_19->addLayout(verticalLayout_11);


        verticalLayout_8->addLayout(horizontalLayout_19);

        scrollArea_2->setWidget(scrollAreaWidgetContents_2);

        verticalLayout_7->addWidget(scrollArea_2);

        tabWidget->addTab(ParticleTab, QString());

        verticalLayout->addWidget(tabWidget);


        retranslateUi(HierarchyViewer);
        QObject::connect(horizontalSlider_EmitAmount, SIGNAL(valueChanged(int)), spinBox_EmitAmount, SLOT(setValue(int)));
        QObject::connect(spinBox_EmitAmount, SIGNAL(valueChanged(int)), horizontalSlider_EmitAmount, SLOT(setValue(int)));

        tabWidget->setCurrentIndex(2);


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
        label_13->setText(QApplication::translate("HierarchyViewer", "Mass", 0));
        label_32->setText(QApplication::translate("HierarchyViewer", "Gravity Factor", 0));
        label_18->setText(QApplication::translate("HierarchyViewer", "Min Life", 0));
        label_14->setText(QApplication::translate("HierarchyViewer", "Size", 0));
        label_17->setText(QApplication::translate("HierarchyViewer", "Emit Amount", 0));
        label_15->setText(QApplication::translate("HierarchyViewer", "Emit Rate", 0));
        label_19->setText(QApplication::translate("HierarchyViewer", "Max Life", 0));
        label_34->setText(QApplication::translate("HierarchyViewer", "Color", 0));
        checkBox_RandomColor->setText(QApplication::translate("HierarchyViewer", "Random", 0));
        label_36->setText(QApplication::translate("HierarchyViewer", "Texture", 0));
        label_20->setText(QApplication::translate("HierarchyViewer", "Force", 0));
        label_21->setText(QApplication::translate("HierarchyViewer", "X", 0));
        label_22->setText(QApplication::translate("HierarchyViewer", "Y", 0));
        label_23->setText(QApplication::translate("HierarchyViewer", "Z", 0));
        label_24->setText(QApplication::translate("HierarchyViewer", "Min Velocity", 0));
        label_25->setText(QApplication::translate("HierarchyViewer", "X", 0));
        label_26->setText(QApplication::translate("HierarchyViewer", "Y", 0));
        label_27->setText(QApplication::translate("HierarchyViewer", "Z", 0));
        label_28->setText(QApplication::translate("HierarchyViewer", "Max Velocity", 0));
        label_29->setText(QApplication::translate("HierarchyViewer", "X", 0));
        label_30->setText(QApplication::translate("HierarchyViewer", "Y", 0));
        label_31->setText(QApplication::translate("HierarchyViewer", "Z", 0));
        tabWidget->setTabText(tabWidget->indexOf(ParticleTab), QApplication::translate("HierarchyViewer", "Particle System", 0));
    } // retranslateUi

};

namespace Ui {
    class HierarchyViewer: public Ui_HierarchyViewer {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_HIERARCHYWIDGET_H
