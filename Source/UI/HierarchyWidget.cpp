#include "HierarchyWidget.h"
#include "ui_HierarchyWidget.h"
#include <Primitives/Puppet.h>

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
	connect(ui->dial_RotationX, SIGNAL(valueChanged(int)), this, SLOT(onRotationXDialChange(int)));
	connect(ui->dial_RotationY, SIGNAL(valueChanged(int)), this, SLOT(onRotationYDialChange(int)));
	connect(ui->dial_RotationZ, SIGNAL(valueChanged(int)), this, SLOT(onRotationZDialChange(int)));

	// popup menu
	m_deleteAction = new QAction("Delete", this);
	ui->treeWidget->setContextMenuPolicy(Qt::CustomContextMenu);
	connect(ui->treeWidget, SIGNAL(customContextMenuRequested(const QPoint)), this, SLOT(showMouseRightButton(const QPoint)));
	connect(m_deleteAction, SIGNAL(triggered()), this, SLOT(deleteGameObject()));

	// tab widget
	particleSystemTab = ui->tabWidget->widget(3);
	ui->tabWidget->removeTab(3);
	ui->tabWidget->setCurrentIndex(0);
	ui->graphicsView_ColorPicker->setScene(new QGraphicsScene(this));
	ui->graphicsView_ParticleTexturePicker->setScene(new QGraphicsScene(this));

	ui->graphicsView_AmbientColorPicker->setScene(new QGraphicsScene(this));
	ui->graphicsView_DiffuseColorPicker->setScene(new QGraphicsScene(this));
	ui->graphicsView_SpecularColorPicker->setScene(new QGraphicsScene(this));
	ui->graphicsView_EmissiveColorPicker->setScene(new QGraphicsScene(this));
	ui->graphicsView_AmbientColorPicker->setBackgroundBrush(QBrush(Qt::black, Qt::DiagCrossPattern));
	ui->graphicsView_DiffuseColorPicker->setBackgroundBrush(QBrush(Qt::white, Qt::DiagCrossPattern));
	ui->graphicsView_SpecularColorPicker->setBackgroundBrush(QBrush(Qt::white, Qt::DiagCrossPattern));
	ui->graphicsView_EmissiveColorPicker->setBackgroundBrush(QBrush(Qt::black, Qt::DiagCrossPattern));
	ui->graphicsView_DiffuseMapPicker->setScene(new QGraphicsScene(this));
	ui->graphicsView_NormalMapPicker->setScene(new QGraphicsScene(this));
	ui->graphicsView_SpecularMapPicker->setScene(new QGraphicsScene(this));
	ui->graphicsView_EmissiveMapPicker->setScene(new QGraphicsScene(this));

	ui->graphicsView_AmbientColorPicker->installEventFilter(this);
	ui->graphicsView_DiffuseColorPicker->installEventFilter(this);
	ui->graphicsView_SpecularColorPicker->installEventFilter(this);
	ui->graphicsView_EmissiveColorPicker->installEventFilter(this);
	ui->graphicsView_NormalMapPicker->installEventFilter(this);

	ui->graphicsView_ColorPicker->installEventFilter(this);
	ui->graphicsView_ParticleTexturePicker->installEventFilter(this);

	connect(ui->doubleSpinBox_Shininess, SIGNAL(valueChanged(double)), this, SLOT(onShininessDoubleBoxChange(double)));
	connect(ui->doubleSpinBox_ShininessStrength, SIGNAL(valueChanged(double)), this, SLOT(onShininessStrengthDoubleBoxChange(double)));
	connect(ui->horizontalSlider_Shininess, SIGNAL(valueChanged(int)), this, SLOT(onShininessSliderChange(int)));
	connect(ui->horizontalSlider_ShininessStrength, SIGNAL(valueChanged(int)), this, SLOT(onShininessStrengthSliderChange(int)));

	connect(ui->checkBox_RandomColor, SIGNAL(toggled(bool)), this, SLOT(setColorPickerEnabled(bool)));
	connect(ui->checkBox_EnableCollision, SIGNAL(toggled(bool)), ui->doubleSpinBox_Restitution, SLOT(setEnabled(bool)));

	setMaximumWidth(360);
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
	if(!current)
	{
		clearTransformationArea();
		return;
	}
	else if (current == ui->treeWidget->topLevelItem(0))
		m_currentObject = m_scene->sceneNode();
	else
		m_currentObject = m_scene->objectManager()->getGameObject(current->text(0)).data();

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
		m_currentObject = NULL;
		m_currentShadingTech = NULL;
		return;
	}

	// get the selected game object
	m_currentObject = m_scene->objectManager()->getGameObject(current->text(0)).data();
	if(!m_currentObject) return;

	// map the transformation into the transform tab
	ui->doubleSpinBox_PositionX->setValue(m_currentObject->position().x());
	ui->doubleSpinBox_PositionY->setValue(m_currentObject->position().y());
	ui->doubleSpinBox_PositionZ->setValue(m_currentObject->position().z());

	ui->doubleSpinBox_RotationX->setValue(m_currentObject->rotation().x());
	ui->doubleSpinBox_RotationY->setValue(m_currentObject->rotation().y());
	ui->doubleSpinBox_RotationZ->setValue(m_currentObject->rotation().z());
	ui->dial_RotationX->setValue((int)m_currentObject->rotation().x());
	ui->dial_RotationY->setValue((int)m_currentObject->rotation().y());
	ui->dial_RotationZ->setValue((int)m_currentObject->rotation().z());
	
	ui->doubleSpinBox_ScaleX->setValue(m_currentObject->scale().x());
	ui->doubleSpinBox_ScaleY->setValue(m_currentObject->scale().y());
	ui->doubleSpinBox_ScaleZ->setValue(m_currentObject->scale().z());

	// map the shading properties into the material tab
	readShadingProperties();
	
	// set connections
	connectCurrentObject();

	// if the game object has a particle system attached to
	// show the particle system tab
	ui->tabWidget->removeTab(ui->tabWidget->indexOf(particleSystemTab));
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
	ui->dial_RotationX->setValue(0);
	ui->dial_RotationY->setValue(0);
	ui->dial_RotationZ->setValue(0);

	ui->doubleSpinBox_ScaleX->setValue(1);
	ui->doubleSpinBox_ScaleY->setValue(1);
	ui->doubleSpinBox_ScaleZ->setValue(1);
}

void HierarchyWidget::connectCurrentObject()
{
	// transformation tab related
	connect(ui->doubleSpinBox_PositionX, SIGNAL(valueChanged(double)), m_currentObject, SLOT(fixedTranslateX(double)));
	connect(ui->doubleSpinBox_PositionY, SIGNAL(valueChanged(double)), m_currentObject, SLOT(fixedTranslateY(double)));
	connect(ui->doubleSpinBox_PositionZ, SIGNAL(valueChanged(double)), m_currentObject, SLOT(fixedTranslateZ(double)));
	connect(ui->doubleSpinBox_RotationX, SIGNAL(valueChanged(double)), m_currentObject, SLOT(fixedRotateX(double)));
	connect(ui->doubleSpinBox_RotationY, SIGNAL(valueChanged(double)), m_currentObject, SLOT(fixedRotateY(double)));
	connect(ui->doubleSpinBox_RotationZ, SIGNAL(valueChanged(double)), m_currentObject, SLOT(fixedRotateZ(double)));
	connect(ui->doubleSpinBox_ScaleX,	 SIGNAL(valueChanged(double)), m_currentObject, SLOT(fixedScaleX(double)));
	connect(ui->doubleSpinBox_ScaleY,	 SIGNAL(valueChanged(double)), m_currentObject, SLOT(fixedScaleY(double)));
	connect(ui->doubleSpinBox_ScaleZ,	 SIGNAL(valueChanged(double)), m_currentObject, SLOT(fixedScaleZ(double)));
	connect(ui->doubleSpinBox_RotationX, SIGNAL(valueChanged(double)), this, SLOT(onRotationXSpinChange(double)));
	connect(ui->doubleSpinBox_RotationY, SIGNAL(valueChanged(double)), this, SLOT(onRotationYSpinChange(double)));
	connect(ui->doubleSpinBox_RotationZ, SIGNAL(valueChanged(double)), this, SLOT(onRotationZSpinChange(double)));

	// puppet tab related
	connect(ui->pushButton_PuppetGo, SIGNAL(clicked()), this, SLOT(assignPuppet()));
	connect(ui->pushButton_PuppetStop, SIGNAL(clicked()), m_currentObject, SLOT(clearPuppets()));

	// rendering tab related
	connect(ui->radioButton_Fill,  SIGNAL(toggled(bool)), m_currentObject, SLOT(toggleFill(bool)));
	connect(ui->radioButton_Line,  SIGNAL(toggled(bool)), m_currentObject, SLOT(toggleWireframe(bool)));
	connect(ui->radioButton_Point, SIGNAL(toggled(bool)), m_currentObject, SLOT(togglePoints(bool)));
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

	// puppet tab related
	disconnect(ui->pushButton_PuppetGo, SIGNAL(clicked()), 0, 0);
	disconnect(ui->pushButton_PuppetStop, SIGNAL(clicked()), 0, 0);

	// rendering tab related
	disconnect(ui->radioButton_Fill,  SIGNAL(toggled(bool)), 0, 0);
	disconnect(ui->radioButton_Line,  SIGNAL(toggled(bool)), 0, 0);
	disconnect(ui->radioButton_Point, SIGNAL(toggled(bool)), 0, 0);

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
	disconnect(ui->checkBox_EnableCollision, SIGNAL(toggled(bool)), 0, 0);
	disconnect(ui->doubleSpinBox_Restitution,	SIGNAL(valueChanged(double)), 0, 0);
}

void HierarchyWidget::renameGameObject( QTreeWidgetItem * item, int column )
{
	// ignore the root node
	if(item == ui->treeWidget->topLevelItem(0)) return;

	// delete the current one
	GameObjectPtr go = m_scene->objectManager()->m_gameObjectMap.take(m_currentObject->objectName());

	// add the new record
	go->setObjectName(item->text(column));
	m_scene->objectManager()->m_gameObjectMap[go->objectName()] = go;
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
	m_scene->objectManager()->deleteObject(m_currentObject->objectName());
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
	connect(ui->checkBox_EnableCollision, SIGNAL(toggled(bool)), ps.data(), SLOT(toggleCollision(bool)));
	connect(ui->doubleSpinBox_Restitution,	 SIGNAL(valueChanged(double)), ps.data(), SLOT(setRestitution(double)));
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
	ui->checkBox_EnableCollision->setChecked(ps->isCollisionEnabled());
	ui->doubleSpinBox_Restitution->setValue(ps->getRestitution());

	// displays the particle texture
	ui->graphicsView_ParticleTexturePicker->scene()->clear();
	QPixmap tex = ps->getTexture()->generateQPixmap();
	QGraphicsPixmapItem* item = new QGraphicsPixmapItem(tex);
	ui->graphicsView_ParticleTexturePicker->scene()->addItem(item);
	ui->graphicsView_ParticleTexturePicker->fitInView(item);
}

void HierarchyWidget::setColorPickerEnabled( bool status )
{
	// if the random check box is checked, disable the color picker
	ui->graphicsView_ColorPicker->setEnabled(!status);
}

bool HierarchyWidget::eventFilter( QObject *obj, QEvent *ev )
{
	// pop up a color dialog when the user clicks the color picker
	if (ev->type() == QEvent::MouseButtonPress)
	{
		if (obj == ui->graphicsView_ColorPicker 
			&& ui->graphicsView_ColorPicker->isEnabled())
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
		// pop up a file dialog when the user clicks the texture picker
		else if (obj == ui->graphicsView_ParticleTexturePicker)
		{
			QString fileName = QFileDialog::getOpenFileName(0, tr("Select a texture"),
				"../Resource/Textures",
				tr("Texture File(*.*)"));
			if (!fileName.isEmpty())
			{
				// apply the texture to the particle system and color picker both
				ParticleSystemPtr ps = m_currentObject->getComponent("ParticleSystem").dynamicCast<ParticleSystem>();
				ps->loadTexture(fileName);

				ui->graphicsView_ParticleTexturePicker->scene()->clear();
				QPixmap tex = ps->getTexture()->generateQPixmap();
				QGraphicsPixmapItem* item = new QGraphicsPixmapItem(tex);
				ui->graphicsView_ParticleTexturePicker->scene()->addItem(item);
				ui->graphicsView_ParticleTexturePicker->fitInView(item);
			}
			return true;
		}
		/// Material section
		else if (obj == ui->graphicsView_AmbientColorPicker)
		{
			if (!m_currentShadingTech) return true;
			ComponentPtr comp = m_currentObject->getComponent("Model");
			ModelPtr model = comp.dynamicCast<AbstractModel>();
			MaterialPtr mat = model->getMaterial();

			QColor col = QColorDialog::getColor(mat->m_ambientColor, this);
			if(col.isValid()) 
			{
				// apply the color to the particle system and color picker both
				ui->graphicsView_AmbientColorPicker->setBackgroundBrush(QBrush(col, Qt::DiagCrossPattern));
				
				// change the material of the model
				mat->m_ambientColor = col;

				// set the diffuse color through the shading effect
				m_currentShadingTech->enable();
				m_currentShadingTech->setMatAmbientColor(col);
			}
			return true;
		}
		else if (obj == ui->graphicsView_DiffuseColorPicker)
		{
			if (!m_currentShadingTech) return true;
			ComponentPtr comp = m_currentObject->getComponent("Model");
			ModelPtr model = comp.dynamicCast<AbstractModel>();
			MaterialPtr mat = model->getMaterial();

			QColor col = QColorDialog::getColor(mat->m_diffuseColor, this);
			if(col.isValid()) 
			{
				// apply the color to the particle system and color picker both
				ui->graphicsView_DiffuseColorPicker->setBackgroundBrush(QBrush(col, Qt::DiagCrossPattern));
				
				// change the material of the model
				mat->m_diffuseColor = col;

				// set the diffuse color through the shading effect
				m_currentShadingTech->enable();
				m_currentShadingTech->setMatDiffuseColor(col);
			}
			return true;
		}
		else if (obj == ui->graphicsView_SpecularColorPicker)
		{
			if (!m_currentShadingTech) return true;
			ComponentPtr comp = m_currentObject->getComponent("Model");
			ModelPtr model = comp.dynamicCast<AbstractModel>();
			MaterialPtr mat = model->getMaterial();

			QColor col = QColorDialog::getColor(mat->m_specularColor, this);
			if(col.isValid()) 
			{
				// apply the color to the particle system and color picker both
				ui->graphicsView_SpecularColorPicker->setBackgroundBrush(QBrush(col, Qt::DiagCrossPattern));
				
				// change the material of the model
				mat->m_specularColor = col;

				// set the diffuse color through the shading effect
				m_currentShadingTech->enable();
				m_currentShadingTech->setMatSpecularColor(col);
			}
			return true;
		}
		else if (obj == ui->graphicsView_EmissiveColorPicker)
		{
			if (!m_currentShadingTech) return true;
			ComponentPtr comp = m_currentObject->getComponent("Model");
			ModelPtr model = comp.dynamicCast<AbstractModel>();
			MaterialPtr mat = model->getMaterial();

			QColor col = QColorDialog::getColor(mat->m_emissiveColor, this);
			if(col.isValid()) 
			{
				// apply the color to the particle system and color picker both
				ui->graphicsView_EmissiveColorPicker->setBackgroundBrush(QBrush(col, Qt::DiagCrossPattern));
				
				// change the material of the model
				mat->m_emissiveColor = col;

				// set the diffuse color through the shading effect
				m_currentShadingTech->enable();
				m_currentShadingTech->setMatEmissiveColor(col);
			}
			return true;
		}
		else if (obj == ui->graphicsView_NormalMapPicker)
		{
			if (!m_currentShadingTech) return true;
			ComponentPtr comp = m_currentObject->getComponent("Model");
			ModelPtr model = comp.dynamicCast<AbstractModel>();
			QString fileName = QFileDialog::getOpenFileName(0, tr("Select a normal map texture"),
				QFileInfo(model->fileName()).absolutePath(),
				tr("Texture File(*.*)"));
			if (!fileName.isEmpty())
			{
				// apply the texture to the particle system and color picker both
				qDebug() << "normal mapping insertion hasn't been implemented yet!";
			}

			return true;
		}
		else
		{
			return QWidget::eventFilter(obj, ev);
		}
	}
	else
	{
		return QWidget::eventFilter(obj, ev);
	}
}

void HierarchyWidget::searchSuitableShaders(ModelPtr currentModel)
{
	// don't apply shader when searching for shaders
	disconnect(ui->comboBox_SahderFiles, 0, 0, 0);
	ui->comboBox_SahderFiles->clear();

	QStringList nameFilter("*.frag");
	QDir dir("../Resource/Shaders/");
	QStringList shaderFiles = dir.entryList(nameFilter);

	// extract the file name and add it to the combo box
	foreach(QString fileName, shaderFiles)
	{
		int dot = fileName.lastIndexOf(".");
		fileName = fileName.left(dot);
		
		if ((currentModel.dynamicCast<StaticModel>()
			&& fileName.contains("static")) || 
			(currentModel.dynamicCast<RiggedModel>()
			&& fileName.contains("skinning")))
		{
			// only add the right types of shaders
			ui->comboBox_SahderFiles->addItem(fileName);
		}
	}
	
	// connect it when the loading is finished
	connect(ui->comboBox_SahderFiles, SIGNAL(currentTextChanged(const QString&)), this, SLOT(changeShader(const QString&)));
}


void HierarchyWidget::changeShader( const QString& shaderFile )
{
	if (!m_currentShadingTech || m_currentShadingTech->shaderFileName() == shaderFile) return;
	m_currentShadingTech->applyShader(shaderFile);

	// re assign the material properties
	ComponentPtr comp = m_currentObject->getComponent("Model");
	ModelPtr model = comp.dynamicCast<AbstractModel>();
	MaterialPtr mat = model->getMaterial();
	m_currentShadingTech->setMaterial(mat);
}


void HierarchyWidget::onShininessSliderChange( int value )
{
	ui->doubleSpinBox_Shininess->setValue(value);
	if (!m_currentShadingTech) return;
	m_currentShadingTech->enable();
	m_currentShadingTech->setMatSpecularPower(value);

	// change the material of the model
	ComponentPtr comp = m_currentObject->getComponent("Model");
	ModelPtr model = comp.dynamicCast<AbstractModel>();
	MaterialPtr mat = model->getMaterial();
	mat->m_shininess = value;
}

void HierarchyWidget::onShininessDoubleBoxChange( double value )
{
	ui->horizontalSlider_Shininess->setValue(value);
	if (!m_currentShadingTech) return;
	m_currentShadingTech->enable();
	m_currentShadingTech->setMatSpecularPower(value);

	// change the material of the model
	ComponentPtr comp = m_currentObject->getComponent("Model");
	ModelPtr model = comp.dynamicCast<AbstractModel>();
	MaterialPtr mat = model->getMaterial();
	mat->m_shininess = value;
}

void HierarchyWidget::onShininessStrengthSliderChange( int value )
{
	ui->doubleSpinBox_ShininessStrength->setValue(value/(double)100);
	if (!m_currentShadingTech) return;
	m_currentShadingTech->enable();
	m_currentShadingTech->setMatSpecularIntensity(value/100.0f);

	// change the material of the model
	ComponentPtr comp = m_currentObject->getComponent("Model");
	ModelPtr model = comp.dynamicCast<AbstractModel>();
	MaterialPtr mat = model->getMaterial();
	mat->m_shininessStrength = value/100.0f;
}

void HierarchyWidget::onShininessStrengthDoubleBoxChange( double value )
{
	ui->horizontalSlider_ShininessStrength->setValue(value * 100);
	if (!m_currentShadingTech) return;
	m_currentShadingTech->enable();
	m_currentShadingTech->setMatSpecularIntensity(value);

	// change the material of the model
	ComponentPtr comp = m_currentObject->getComponent("Model");
	ModelPtr model = comp.dynamicCast<AbstractModel>();
	MaterialPtr mat = model->getMaterial();
	mat->m_shininessStrength = value;
}

void HierarchyWidget::readShadingProperties()
{
	ui->graphicsView_DiffuseMapPicker->scene()->clear();
	ui->graphicsView_NormalMapPicker->scene()->clear();
	ComponentPtr comp = m_currentObject->getComponent("Model");
	ModelPtr model = comp.dynamicCast<AbstractModel>();
	if (!model) return;
	searchSuitableShaders(model);

	m_currentShadingTech = model->renderingEffect().data();
	if (!m_currentShadingTech) return;

	ui->comboBox_SahderFiles->setCurrentText(m_currentShadingTech->shaderFileName());

	MaterialPtr mat = model->getMaterial();
	// map the material information into the tab
	ui->graphicsView_AmbientColorPicker->setBackgroundBrush(QBrush(mat->m_ambientColor, Qt::DiagCrossPattern));
	ui->graphicsView_DiffuseColorPicker->setBackgroundBrush(QBrush(mat->m_diffuseColor, Qt::DiagCrossPattern));
	ui->graphicsView_SpecularColorPicker->setBackgroundBrush(QBrush(mat->m_specularColor, Qt::DiagCrossPattern));
	ui->graphicsView_EmissiveColorPicker->setBackgroundBrush(QBrush(mat->m_emissiveColor, Qt::DiagCrossPattern));

	ui->doubleSpinBox_Shininess->setValue(mat->m_shininess);
	ui->doubleSpinBox_ShininessStrength->setValue(mat->m_shininessStrength);

	// map the textures
	
}

void HierarchyWidget::onRotationXDialChange( int val )
{
	ui->doubleSpinBox_RotationX->setValue((double)val);
}

void HierarchyWidget::onRotationYDialChange( int val )
{
	ui->doubleSpinBox_RotationY->setValue((double)val);
}

void HierarchyWidget::onRotationZDialChange( int val )
{
	ui->doubleSpinBox_RotationZ->setValue((double)val);
}

void HierarchyWidget::onRotationXSpinChange( double val )
{
	ui->dial_RotationX->setValue((int)val);
}

void HierarchyWidget::onRotationYSpinChange( double val )
{
	ui->dial_RotationY->setValue((int)val);
}

void HierarchyWidget::onRotationZSpinChange( double val )
{
	ui->dial_RotationZ->setValue((int)val);
}

void HierarchyWidget::assignPuppet()
{
	Puppet::Variable type;
	if (ui->comboBox_BehaviourType->currentText() == "Position")
		type = Puppet::Position;
	else if (ui->comboBox_BehaviourType->currentText() == "Rotation")
		type = Puppet::Rotation;
	else if (ui->comboBox_BehaviourType->currentText() == "Scale")
		type = Puppet::Scale;

	vec3 speed(ui->doubleSpinBox_PuppetSpeedX->value(),
			   ui->doubleSpinBox_PuppetSpeedY->value(),
			   ui->doubleSpinBox_PuppetSpeedZ->value());

	float duration = ui->doubleSpinBox_PuppetDuration->value();

	// only active a puppet when the speed is not 0
	if (!speed.isNull())
		// this instance will be removed automatically
		Puppet* pup = new Puppet(m_currentObject, type, speed, duration);
}
