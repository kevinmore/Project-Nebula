#include "HierarchyWidget.h"
#include "ui_HierarchyWidget.h"

HierarchyWidget::HierarchyWidget(Scene* scene, QWidget *parent)
	: QWidget(parent),
	  m_scene(scene),
	  m_currentObject(0),
	  ui(new Ui::HierarchyViewer)

{
	ui->setupUi(this);
	// tree widget related
	connect(m_scene, SIGNAL(updateHierarchy()), this, SLOT(updateObjectTree()));
	connect(ui->treeWidget, SIGNAL(currentItemChanged(QTreeWidgetItem*, QTreeWidgetItem*)), 
		    this, SLOT(updateTransformation(QTreeWidgetItem*, QTreeWidgetItem*)));

	// reset button
	connect(ui->pushButton_Reset, SIGNAL(clicked()), this, SLOT(resetSelectedObject()));
	
	updateObjectTree();
}

HierarchyWidget::~HierarchyWidget()
{
	delete ui;
}

void HierarchyWidget::updateObjectTree()
{
	ui->treeWidget->clear();
	readHierarchy(m_scene->sceneNode(), 0);
	ui->treeWidget->expandAll();
}

void HierarchyWidget::readHierarchy( GameObject* go, QTreeWidgetItem* parentItem )
{
	if(!go) return;
	QTreeWidgetItem* item;
	if(parentItem)
	{
		item = new QTreeWidgetItem(parentItem);
	}
	else
	{
		item = new QTreeWidgetItem(ui->treeWidget);
	}
	item->setText(0, go->objectName());

	foreach(QObject* obj, go->children())
	{
		readHierarchy((GameObject*)obj, item);
	}
}

void HierarchyWidget::resetHierarchy( GameObject* go )
{
	if (!go) return;

	go->reset();
//////////////////////////////////////////////////////////////////////////
	// BUG BELOW!!!!!!!!!!!!!!!!!!
// 	foreach(QObject* obj, go->children())
// 	{
// 		resetHierarchy((GameObject*)obj);
// 	}
}

void HierarchyWidget::resetSelectedObject()
{
	// get the selected game object
	QTreeWidgetItem* current = ui->treeWidget->currentItem();
	if(!current) return;
	else if (current == ui->treeWidget->topLevelItem(0))
		m_currentObject = m_scene->sceneNode();
	else
		m_currentObject = m_scene->modelManager()->getModel(current->text(0))->gameObject();

	clearTransformationArea();
	resetHierarchy(m_currentObject);
}

void HierarchyWidget::updateTransformation(QTreeWidgetItem* current, QTreeWidgetItem* previous)
{
	if (!current) return;

	// disconnect previous connections
	disconnectPreviousObject();

	// if the current item is the scene node (root), ignore
	if(current == ui->treeWidget->topLevelItem(0)) 
	{
		clearTransformationArea();
		return;
	}

	// get the selected game object
	m_currentObject = m_scene->modelManager()->getModel(current->text(0))->gameObject();

	// map the transformation into the spin boxes
	ui->doubleSpinBox_PositionX->setValue(m_currentObject->position().x());
	ui->doubleSpinBox_PositionY->setValue(m_currentObject->position().y());
	ui->doubleSpinBox_PositionZ->setValue(m_currentObject->position().z());

	ui->doubleSpinBox_RotationX->setValue(m_currentObject->rotation().x());
	ui->doubleSpinBox_RotationY->setValue(m_currentObject->rotation().y());
	ui->doubleSpinBox_RotationZ->setValue(m_currentObject->rotation().z());

	ui->doubleSpinBox_ScaleX->setValue(m_currentObject->scale().x());
	ui->doubleSpinBox_ScaleY->setValue(m_currentObject->scale().y());
	ui->doubleSpinBox_ScaleZ->setValue(m_currentObject->scale().z());

	// set connections
	connectCurrentObject();
}

void HierarchyWidget::clearTransformationArea()
{
	ui->doubleSpinBox_PositionX->setValue(0);
	ui->doubleSpinBox_PositionY->setValue(0);
	ui->doubleSpinBox_PositionZ->setValue(0);

	ui->doubleSpinBox_RotationX->setValue(0);
	ui->doubleSpinBox_RotationY->setValue(0);
	ui->doubleSpinBox_RotationZ->setValue(0);

	ui->doubleSpinBox_ScaleX->setValue(1);
	ui->doubleSpinBox_ScaleY->setValue(1);
	ui->doubleSpinBox_ScaleZ->setValue(1);
}

void HierarchyWidget::connectCurrentObject()
{
	// transformation panel related
	connect(ui->doubleSpinBox_PositionX, SIGNAL(valueChanged(double)), m_currentObject, SLOT(translateX(double)));
	connect(ui->doubleSpinBox_PositionY, SIGNAL(valueChanged(double)), m_currentObject, SLOT(translateY(double)));
	connect(ui->doubleSpinBox_PositionZ, SIGNAL(valueChanged(double)), m_currentObject, SLOT(translateZ(double)));
	connect(ui->doubleSpinBox_RotationX, SIGNAL(valueChanged(double)), m_currentObject, SLOT(rotateX(double)));
	connect(ui->doubleSpinBox_RotationY, SIGNAL(valueChanged(double)), m_currentObject, SLOT(rotateY(double)));
	connect(ui->doubleSpinBox_RotationZ, SIGNAL(valueChanged(double)), m_currentObject, SLOT(rotateZ(double)));
	connect(ui->doubleSpinBox_ScaleX,	 SIGNAL(valueChanged(double)), m_currentObject, SLOT(scaleX(double)));
	connect(ui->doubleSpinBox_ScaleY,	 SIGNAL(valueChanged(double)), m_currentObject, SLOT(scaleY(double)));
	connect(ui->doubleSpinBox_ScaleZ,	 SIGNAL(valueChanged(double)), m_currentObject, SLOT(scaleZ(double)));
}

void HierarchyWidget::disconnectPreviousObject()
{
	// transformation panel related
	disconnect(ui->doubleSpinBox_PositionX, SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_PositionY, SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_PositionZ, SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_RotationX, SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_RotationY, SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_RotationZ, SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_ScaleX,		SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_ScaleY,		SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_ScaleZ,		SIGNAL(valueChanged(double)), 0, 0);
}
