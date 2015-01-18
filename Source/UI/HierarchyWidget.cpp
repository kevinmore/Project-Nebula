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
		    this, SLOT(readGameObject(QTreeWidgetItem*, QTreeWidgetItem*)));

	connect(ui->treeWidget, SIGNAL(itemChanged(QTreeWidgetItem*, int)), 
		    this, SLOT(renameGameObject(QTreeWidgetItem*, int)));
	
	// transform reset button
	connect(ui->pushButton_Reset, SIGNAL(clicked()), this, SLOT(resetSelectedObject()));
	
	// popup menu
	m_deleteAction = new QAction("Delete", this);
	ui->treeWidget->setContextMenuPolicy(Qt::CustomContextMenu);
	connect(ui->treeWidget, SIGNAL(customContextMenuRequested(const QPoint)), this, SLOT(showMouseRightButton(const QPoint)));
	connect(m_deleteAction, SIGNAL(triggered()), this, SLOT(deleteGameObject()));

	// tab widget
	particleSystemTab = ui->tabWidget->widget(2);
	ui->tabWidget->removeTab(2);
	ui->tabWidget->setCurrentIndex(0);
	ui->graphicsView_ColorPicker->setScene(new QGraphicsScene(this));
	ui->graphicsView_ColorPicker->installEventFilter(this);
	connect(ui->checkBox_RandomColor, SIGNAL(toggled(bool)), this, SLOT(setColorPickerEnabled(bool)));

	setMaximumWidth(345);
	updateObjectTree();
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
		item->setFlags(item->flags() | Qt::ItemIsEditable);
	}
	else
	{
		item = new QTreeWidgetItem(ui->treeWidget);
	}

	// each item displays the name of the game object
	// each item should be editable
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

	foreach(QObject* obj, go->children())
	{
		resetHierarchy((GameObject*)obj);
	}
}

void HierarchyWidget::resetSelectedObject()
{
	// get the selected game object
	QTreeWidgetItem* current = ui->treeWidget->currentItem();
	if(!current) return;
	else if (current == ui->treeWidget->topLevelItem(0))
		m_currentObject = m_scene->sceneNode();
	else
		m_currentObject = m_scene->modelManager()->getGameObject(current->text(0)).data();

	clearTransformationArea();
	resetHierarchy(m_currentObject);
}

void HierarchyWidget::readGameObject(QTreeWidgetItem* current, QTreeWidgetItem* previous)
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
	m_currentObject = m_scene->modelManager()->getGameObject(current->text(0)).data();
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

	// if the game object has a particle system attached to
	// show the particle system tab
	ui->tabWidget->removeTab(ui->tabWidget->indexOf(particleSystemTab));
	ui->tabWidget->setCurrentIndex(0);
	foreach(ComponentPtr comp, m_currentObject->getComponents())
	{
		if (comp->className() == "ParticleSystem")
		{
			ParticleSystemPtr ps = comp.dynamicCast<ParticleSystem>();
			ui->tabWidget->addTab(particleSystemTab, "Particle System");
			ui->tabWidget->setCurrentWidget(particleSystemTab);
			readParticleSystemConfig(ps);
			connectParticleSystemTab(ps);
			
			break;
		}
	}
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
	// transformation tab related
	disconnect(ui->doubleSpinBox_PositionX, SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_PositionY, SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_PositionZ, SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_RotationX, SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_RotationY, SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_RotationZ, SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_ScaleX,		SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_ScaleY,		SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_ScaleZ,		SIGNAL(valueChanged(double)), 0, 0);

	// particle system tab related
	disconnect(ui->doubleSpinBox_Mass,			SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_GravityFactor,	SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_Size,			SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_EmitRate,		SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->spinBox_EmitAmount,			SIGNAL(valueChanged(int)),	  0, 0);
	disconnect(ui->horizontalSlider_EmitAmount,	SIGNAL(valueChanged(int)),	  0, 0);
	disconnect(ui->doubleSpinBox_MinLife	,		SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_MaxLife	,		SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_ForceX,			SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_ForceY,			SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_ForceZ,			SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_MinVelocityX,	SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_MinVelocityY,	SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_MinVelocityZ,	SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_MaxVelocityX,	SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_MaxVelocityY,	SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_MaxVelocityZ,	SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->checkBox_RandomColor,			SIGNAL(toggled(bool)),		  0, 0);
}

void HierarchyWidget::renameGameObject( QTreeWidgetItem * item, int column )
{
	// ignore the root node
	if(item == ui->treeWidget->topLevelItem(0)) return;

	// delete the current one
	GameObjectPtr go = m_scene->modelManager()->m_gameObjectMap.take(m_currentObject->objectName());

	// add the new record
	go->setObjectName(item->text(column));
	m_scene->modelManager()->m_gameObjectMap[go->objectName()] = go;
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
	GameObjectPtr go = m_scene->modelManager()->m_gameObjectMap.take(m_currentObject->objectName());
	go.clear();

	updateObjectTree();
}

void HierarchyWidget::connectParticleSystemTab(ParticleSystemPtr ps)
{
	// particle system tab related
	connect(ui->doubleSpinBox_Mass,			 SIGNAL(valueChanged(double)), ps.data(), SLOT(setParticleMass(double)));
	connect(ui->doubleSpinBox_GravityFactor,	 SIGNAL(valueChanged(double)), ps.data(), SLOT(setGravityFactor(double)));
	connect(ui->doubleSpinBox_Size,			 SIGNAL(valueChanged(double)), ps.data(), SLOT(setParticleSize(double)));
	connect(ui->doubleSpinBox_EmitRate,		 SIGNAL(valueChanged(double)), ps.data(), SLOT(setEmitRate(double)));
	connect(ui->spinBox_EmitAmount,			 SIGNAL(valueChanged(int)),	   ps.data(), SLOT(setEmitAmount(int)));
	connect(ui->horizontalSlider_EmitAmount,	 SIGNAL(valueChanged(int)),	   ps.data(), SLOT(setEmitAmount(int)));
	connect(ui->doubleSpinBox_MinLife,		 SIGNAL(valueChanged(double)), ps.data(), SLOT(setMinLife(double)));
	connect(ui->doubleSpinBox_MaxLife,		 SIGNAL(valueChanged(double)), ps.data(), SLOT(setMaxLife(double)));
	connect(ui->doubleSpinBox_ForceX,		 SIGNAL(valueChanged(double)), ps.data(), SLOT(setForceX(double)));
	connect(ui->doubleSpinBox_ForceY,		 SIGNAL(valueChanged(double)), ps.data(), SLOT(setForceY(double)));
	connect(ui->doubleSpinBox_ForceZ,		 SIGNAL(valueChanged(double)), ps.data(), SLOT(setForceZ(double)));
	connect(ui->doubleSpinBox_MinVelocityX,	 SIGNAL(valueChanged(double)), ps.data(), SLOT(setMinVelX(double)));
	connect(ui->doubleSpinBox_MinVelocityY,	 SIGNAL(valueChanged(double)), ps.data(), SLOT(setMinVelY(double)));
	connect(ui->doubleSpinBox_MinVelocityZ,	 SIGNAL(valueChanged(double)), ps.data(), SLOT(setMinVelZ(double)));
	connect(ui->doubleSpinBox_MaxVelocityX,	 SIGNAL(valueChanged(double)), ps.data(), SLOT(setMaxVelX(double)));
	connect(ui->doubleSpinBox_MaxVelocityY,	 SIGNAL(valueChanged(double)), ps.data(), SLOT(setMaxVelY(double)));
	connect(ui->doubleSpinBox_MaxVelocityZ,	 SIGNAL(valueChanged(double)), ps.data(), SLOT(setMaxVelZ(double)));
	connect(ui->spinBox_EmitAmount,			 SIGNAL(valueChanged(int)),	   ui->horizontalSlider_EmitAmount, SLOT(setValue(int)));
	connect(ui->horizontalSlider_EmitAmount,	 SIGNAL(valueChanged(int)),	   ui->spinBox_EmitAmount, SLOT(setValue(int)));
	connect(ui->checkBox_RandomColor, SIGNAL(toggled(bool)), ps.data(), SLOT(toggleRandomColor(bool)));
}

void HierarchyWidget::readParticleSystemConfig( ParticleSystemPtr ps )
{
	// map the particle system configurations into the spin boxes
	ui->doubleSpinBox_Mass->setValue(ps->getParticleMass());
	ui->doubleSpinBox_GravityFactor->setValue(ps->getGravityFactor());
	ui->doubleSpinBox_Size->setValue(ps->getParticleSize());
	ui->doubleSpinBox_EmitRate->setValue(ps->getEmitRate());
	ui->horizontalSlider_EmitAmount->setValue(ps->getEmitAmount());
	ui->spinBox_EmitAmount->setValue(ps->getEmitAmount());
	ui->doubleSpinBox_MinLife->setValue(ps->getMinLife());
	ui->doubleSpinBox_MaxLife->setValue(ps->getMaxLife());
	ui->doubleSpinBox_ForceX->setValue(ps->getForce().x());
	ui->doubleSpinBox_ForceY->setValue(ps->getForce().y());
	ui->doubleSpinBox_ForceZ->setValue(ps->getForce().z());
	ui->doubleSpinBox_MinVelocityX->setValue(ps->getMinVel().x());
	ui->doubleSpinBox_MinVelocityY->setValue(ps->getMinVel().y());
	ui->doubleSpinBox_MinVelocityZ->setValue(ps->getMinVel().z());
	ui->doubleSpinBox_MaxVelocityX->setValue(ps->getMaxVel().x());
	ui->doubleSpinBox_MaxVelocityY->setValue(ps->getMaxVel().y());
	ui->doubleSpinBox_MaxVelocityZ->setValue(ps->getMaxVel().z());
	ui->graphicsView_ColorPicker->setBackgroundBrush(QBrush(ps->getParticleColor(), Qt::DiagCrossPattern));
	ui->checkBox_RandomColor->setChecked(ps->isColorRandom());
}

void HierarchyWidget::setColorPickerEnabled( bool status )
{
	// if the random check box is checked, disable the color picker
	ui->graphicsView_ColorPicker->setEnabled(!status);
}

bool HierarchyWidget::eventFilter( QObject *obj, QEvent *ev )
{
	// pop up a color dialog when the user clicks the picker
	if (obj == ui->graphicsView_ColorPicker 
		&& ui->graphicsView_ColorPicker->isEnabled()
		&& ev->type() == QEvent::MouseButtonPress)
	{
		ParticleSystemPtr ps = m_currentObject->getComponent("ParticleSystem").dynamicCast<ParticleSystem>();
		QColor col = QColorDialog::getColor(ps->getParticleColor(), this);
		if(col.isValid()) 
		{
			// apply the color to the particle system and color picker both
			ps->setParticleColor(col);
			ui->graphicsView_ColorPicker->setBackgroundBrush(QBrush(col, Qt::DiagCrossPattern));
		}
		return true;
	}
	else
	{
		return QWidget::eventFilter(obj, ev);
	}
}
