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

	connect(ui->treeWidget, SIGNAL(itemChanged(QTreeWidgetItem*, int)), 
		    this, SLOT(renameGameObject(QTreeWidgetItem*, int)));
	
	// reset button
	connect(ui->pushButton_Reset, SIGNAL(clicked()), this, SLOT(resetSelectedObject()));
	
	// popup menu
	m_deleteAction = new QAction("Delete", this);
	ui->treeWidget->setContextMenuPolicy(Qt::CustomContextMenu);
	connect(ui->treeWidget, SIGNAL(customContextMenuRequested(const QPoint)), this, SLOT(showMouseRightButton(const QPoint)));
	connect(m_deleteAction, SIGNAL(triggered()), this, SLOT(deleteGameObject()));

	updateObjectTree();
	QTreeWidgetItem* root = ui->treeWidget->topLevelItem(0);
	root->setFlags(root->flags() & ~Qt::ItemIsEditable);
}

HierarchyWidget::~HierarchyWidget()
{
	delete ui;
}

void HierarchyWidget::updateObjectTree()
{
	// block the signals emitted from the tree when updating
	// to avoid emitting itemChanged signal
	ui->treeWidget->blockSignals(true);
	ui->treeWidget->clear();
	readHierarchy(m_scene->sceneNode(), 0);
	ui->treeWidget->expandAll();
	ui->treeWidget->blockSignals(false);
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

	// each item displays the name of the game object
	// each item should be editable
	item->setText(0, go->objectName());
	item->setFlags(item->flags() | Qt::ItemIsEditable);

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
		m_currentObject = m_scene->modelManager()->getGameObject(current->text(0));

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
	m_currentObject = m_scene->modelManager()->getGameObject(current->text(0));
	if(!m_currentObject) return;

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

void HierarchyWidget::renameGameObject( QTreeWidgetItem * item, int column )
{
	// ignore the root node
	if(item == ui->treeWidget->topLevelItem(0)) return;

	// delete the current one
	m_scene->modelManager()->m_gameObjectMap.take(m_currentObject->objectName());

	// add the new record
	m_currentObject->setObjectName(item->text(column));
	m_scene->modelManager()->m_gameObjectMap[m_currentObject->objectName()] = m_currentObject;
}

void HierarchyWidget::showMouseRightButton( const QPoint& point )
{
	QTreeWidgetItem* selected = ui->treeWidget->itemAt(point);
	if(!selected || selected == ui->treeWidget->topLevelItem(0)) return;

	QMenu* popMenu = new QMenu(ui->treeWidget);
	popMenu->addAction(m_deleteAction);
	popMenu->exec(QCursor::pos());
}

void HierarchyWidget::deleteGameObject()
{
	// take the object from the map, and delete it
	ModelPtr model = m_scene->modelManager()->m_modelMap.take(m_currentObject->objectName());
	if(model) model.clear();
	else
	{
		m_scene->modelManager()->m_gameObjectMap.take(m_currentObject->objectName());
		SAFE_DELETE(m_currentObject);
	}

	updateObjectTree();
}
