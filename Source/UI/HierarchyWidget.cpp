#include "HierarchyWidget.h"
#include "ui_HierarchyWidget.h"
#include <Primitives/Puppet.h>
#include <Physicis/Collision/Collider/BoxCollider.h>
#include <Physicis/Collision/Collider/SphereCollider.h>

HierarchyWidget::HierarchyWidget(Canvas* canvas, QWidget *parent)
	: QWidget(parent),
	  m_scene(Scene::instance()),
	  m_canvas(canvas),
	  m_currentObject(0),
	  ui(new Ui::HierarchyViewer)

{
	setMaximumWidth(380);

	ui->setupUi(this);
	ui->treeWidget->setContainerWidget(this);

	// connection to a ray casting function in the scene
	connect(m_canvas, SIGNAL(objectPicked(GameObjectPtr)), this, SLOT(onObjectPicked(GameObjectPtr)));
	
	connect(m_scene, SIGNAL(cleared()), this, SLOT(clearReference()));

	// connection from canvas to delete object
	connect(m_canvas, SIGNAL(deleteObject()), this, SLOT(deleteGameObject()));

	// material change behaviour
	connect(this, SIGNAL(materialChanged()), this, SLOT(assignMaterial()));

	// tree widget related
	connect(m_scene, SIGNAL(updateHierarchy()), this, SLOT(updateObjectTree()));

	connect(ui->treeWidget, SIGNAL(currentItemChanged(QTreeWidgetItem*, QTreeWidgetItem*)), 
		    this, SLOT(onSelectedGameObjectChanged(QTreeWidgetItem*, QTreeWidgetItem*)));

	connect(ui->treeWidget, SIGNAL(itemChanged(QTreeWidgetItem*, int)), 
		    this, SLOT(renameGameObject(QTreeWidgetItem*, int)));
	
	// transform reset button
	connect(ui->pushButton_Reset, SIGNAL(clicked()), this, SLOT(resetSelectedObject()));
	connect(ui->dial_RotationX, SIGNAL(valueChanged(int)), this, SLOT(onRotationXDialChange(int)));
	connect(ui->dial_RotationY, SIGNAL(valueChanged(int)), this, SLOT(onRotationYDialChange(int)));
	connect(ui->dial_RotationZ, SIGNAL(valueChanged(int)), this, SLOT(onRotationZDialChange(int)));

	// popup menu
	m_deleteAction = new QAction("Delete", this);
	m_addRigidBodyAction = new QAction("Rigid Body", this);

	ui->treeWidget->setContextMenuPolicy(Qt::CustomContextMenu);
	connect(ui->treeWidget, SIGNAL(customContextMenuRequested(const QPoint)), this, SLOT(showMouseRightButton(const QPoint)));
	connect(m_deleteAction, SIGNAL(triggered()), this, SLOT(deleteGameObject()));
	connect(m_addRigidBodyAction, SIGNAL(triggered()), this, SLOT(createRigidBody()));

	// tab widget
	int tabCount = ui->tabWidget->count();
	m_renderingTab = ui->tabWidget->widget(2);
	m_particleSystemTab = ui->tabWidget->widget(3);
	m_lightTab = ui->tabWidget->widget(4);
	m_rigidBodyTab = ui->tabWidget->widget(5);

	// remove the unnecessary tabs
	for (int i = 0; i < tabCount - 2; ++i)
	{
		ui->tabWidget->removeTab(2);
	}

	ui->tabWidget->setCurrentIndex(0);
	ui->graphicsView_ColorPicker->setScene(new QGraphicsScene(this));
	ui->graphicsView_ParticleTexturePicker->setScene(new QGraphicsScene(this));

	ui->graphicsView_AmbientColorPicker->setScene(new QGraphicsScene(this));
	ui->graphicsView_DiffuseColorPicker->setScene(new QGraphicsScene(this));
	ui->graphicsView_SpecularColorPicker->setScene(new QGraphicsScene(this));
	ui->graphicsView_EmissiveColorPicker->setScene(new QGraphicsScene(this));

	ui->graphicsView_DiffuseMapPicker->setScene(new QGraphicsScene(this));
	ui->graphicsView_NormalMapPicker->setScene(new QGraphicsScene(this));
	ui->graphicsView_SpecularMapPicker->setScene(new QGraphicsScene(this));
	ui->graphicsView_EmissiveMapPicker->setScene(new QGraphicsScene(this));

	ui->graphicsView_LightColorPicker->setScene(new QGraphicsScene(this));

	ui->graphicsView_AmbientColorPicker->installEventFilter(this);
	ui->graphicsView_DiffuseColorPicker->installEventFilter(this);
	ui->graphicsView_SpecularColorPicker->installEventFilter(this);
	ui->graphicsView_EmissiveColorPicker->installEventFilter(this);
	ui->graphicsView_NormalMapPicker->installEventFilter(this);

	ui->graphicsView_ColorPicker->installEventFilter(this);
	ui->graphicsView_ParticleTexturePicker->installEventFilter(this);

	ui->graphicsView_LightColorPicker->installEventFilter(this);

	connect(ui->doubleSpinBox_Shininess, SIGNAL(valueChanged(double)), this, SLOT(onShininessDoubleBoxChange(double)));
	connect(ui->doubleSpinBox_ShininessStrength, SIGNAL(valueChanged(double)), this, SLOT(onShininessStrengthDoubleBoxChange(double)));

	connect(ui->horizontalSlider_Shininess, SIGNAL(valueChanged(int)), this, SLOT(onShininessSliderChange(int)));
	connect(ui->horizontalSlider_ShininessStrength, SIGNAL(valueChanged(int)), this, SLOT(onShininessStrengthSliderChange(int)));

	connect(ui->horizontalSlider_Roughness, SIGNAL(valueChanged(int)), this, SLOT(onRoughnessSliderChange(int)));
	connect(ui->doubleSpinBox_Roughness, SIGNAL(valueChanged(double)), this, SLOT(onRoughnessDoubleBoxChange(double)));

	connect(ui->horizontalSlider_fresnelReflectance, SIGNAL(valueChanged(int)), this, SLOT(onFresnelReflectanceSliderChange(int)));
	connect(ui->doubleSpinBox_fresnelReflectance, SIGNAL(valueChanged(double)), this, SLOT(onFresnelReflectanceDoubleBoxChange(double)));

	connect(ui->doubleSpinBox_refractiveIndex, SIGNAL(valueChanged(double)), this, SLOT(onRefractiveIndexDoubleBoxChange(double)));

	connect(ui->horizontalSlider_LightAttConst, SIGNAL(valueChanged(int)), this, SLOT(onConstantAttenuationSliderChange(int)));
	connect(ui->doubleSpinBox_LightAttConst, SIGNAL(valueChanged(double)), this, SLOT(onConstantAttenuationDoubleBoxChange(double)));

	connect(ui->horizontalSlider_LightAttLinear, SIGNAL(valueChanged(int)), this, SLOT(onLinearAttenuationSliderChange(int)));
	connect(ui->doubleSpinBox_LightAttLinear, SIGNAL(valueChanged(double)), this, SLOT(onLinearAttenuationDoubleBoxChange(double)));

	connect(ui->horizontalSlider_LightAttQuad, SIGNAL(valueChanged(int)), this, SLOT(onQuadraticAttenuationSliderChange(int)));
	connect(ui->doubleSpinBox_LightAttQuad, SIGNAL(valueChanged(double)), this, SLOT(onQuadraticAttenuationDoubleBoxChange(double)));

	connect(ui->horizontalSlider_LightIntensity, SIGNAL(valueChanged(int)), this, SLOT(onLightIntensitySliderChange(int)));
	connect(ui->doubleSpinBox_LightIntensity, SIGNAL(valueChanged(double)), this, SLOT(onLightIntensityDoubleBoxChange(double)));

	connect(ui->horizontalSlider_RigidBodyRestitution, SIGNAL(valueChanged(int)), this, SLOT(onRigidBodyRestitutionSliderChange(int)));
	connect(ui->doubleSpinBox_RigidBodyRestitution, SIGNAL(valueChanged(double)), this, SLOT(onRigidBodyRestitutionDoubleBoxChange(double)));

	connect(ui->checkBox_RandomColor, SIGNAL(toggled(bool)), this, SLOT(setColorPickerEnabled(bool)));
	connect(ui->checkBox_EnableCollision, SIGNAL(toggled(bool)), ui->doubleSpinBox_Restitution, SLOT(setEnabled(bool)));

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
	readHierarchy(m_scene->sceneRoot(), 0);
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
		m_currentObject = m_scene->sceneRoot();
	else
		m_currentObject = m_scene->objectManager()->getGameObject(current->text(0)).data();

	clearTransformationArea();
	resetHierarchy(m_currentObject);
}

void HierarchyWidget::readCurrentGameObject()
{
	// reset the tab widget
	ui->tabWidget->removeTab(ui->tabWidget->indexOf(m_renderingTab));
	ui->tabWidget->removeTab(ui->tabWidget->indexOf(m_particleSystemTab));
	ui->tabWidget->removeTab(ui->tabWidget->indexOf(m_lightTab));
	ui->tabWidget->removeTab(ui->tabWidget->indexOf(m_rigidBodyTab));

	// disconnect previous connections
	disconnectPreviousObject();

	// hide all bounding volumes
	m_scene->toggleDebugMode(false);
	m_currentMaterials.clear();

	// map the transformation into the transform tab
	fillInTransformTab();

	// process components and set up tabs
	QStringList componentTypes = m_currentObject->getComponentsTypes();

	foreach(QString type, componentTypes)
	{
		ComponentPtr comp = m_currentObject->getComponent(type);

		if (type == "Model")
		{
			ui->tabWidget->addTab(m_renderingTab, QIcon("../Resource/StyleSheets/Icons/circle-icons/full-color/flower.png"), "Mesh");

			// show the bounding box
			ModelPtr model = comp.dynamicCast<IModel>();
			if(model && model->getBoundingBox()) model->showBoundingVolume();

			// get the materials of the model
			QVector<MaterialPtr> mats = model->getMaterials();
			foreach(MaterialPtr mat, mats)
			{
				m_currentMaterials << mat.data();
			}

			// map the shading properties into the material tab
			readShadingProperties();
		}
		else if (type == "ParticleSystem")
		{
			ui->tabWidget->addTab(m_particleSystemTab, QIcon("../Resource/StyleSheets/Icons/circle-icons/full-color/colorwheel.png"), "Particle System");
			ui->tabWidget->setCurrentWidget(m_particleSystemTab);
			ParticleSystemPtr ps = comp.dynamicCast<ParticleSystem>();
			readParticleSystemConfig(ps);
			connectParticleSystemTab(ps);
		}
		else if (type == "Light")
		{
			ui->tabWidget->addTab(m_lightTab, QIcon("../Resource/StyleSheets/Icons/circle-icons/full-color/lightbulb.png"), "Light");
			readLightSourceProperties(comp.dynamicCast<Light>());
		}
		else if (type == "RigidBody")
		{
			ui->tabWidget->addTab(m_rigidBodyTab, QIcon("../Resource/StyleSheets/Icons/circle-icons/full-color/trends.png"), "Rigid Body");
			RigidBodyPtr rb = comp.dynamicCast<RigidBody>();
			readRigidBodyProperties(rb);
			connectRigidBodyTab(rb);
		}
	}

	// set connections
	connectCurrentObject();
}


void HierarchyWidget::onSelectedGameObjectChanged(QTreeWidgetItem* current, QTreeWidgetItem* previous)
{
	if (!current) return;

	// if the current item is the scene node (root), ignore
	if(current == ui->treeWidget->topLevelItem(0)) 
	{
		clearTransformationArea();
		clearReference();

		return;
	}
	
	// get the selected game object
	m_currentObject = m_scene->objectManager()->getGameObject(current->text(0)).data();
	if(!m_currentObject) return;

	// read this game object
	readCurrentGameObject();
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
	ui->doubleSpinBox_ScaleFactor->setValue(1);
}

void HierarchyWidget::connectCurrentObject()
{
	// transformation tab related
	connectTransformTab();

	// puppet tab related
	connect(ui->pushButton_PuppetGo, SIGNAL(clicked()), this, SLOT(assignPuppet()));
	connect(ui->pushButton_PuppetStop, SIGNAL(clicked()), m_currentObject, SLOT(clearPuppets()));

	// rendering tab related
	connect(ui->radioButton_Fill,  SIGNAL(toggled(bool)), m_currentObject, SLOT(toggleFill(bool)));
	connect(ui->radioButton_Line,  SIGNAL(toggled(bool)), m_currentObject, SLOT(toggleWireframe(bool)));
	connect(ui->radioButton_Point, SIGNAL(toggled(bool)), m_currentObject, SLOT(togglePoints(bool)));
	connect(ui->checkBox_DiffuseMap, SIGNAL(toggled(bool)), this, SLOT(toggleDiffuseMap(bool)));
	connect(ui->checkBox_NormalMap, SIGNAL(toggled(bool)), this, SLOT(toggleNormalMap(bool)));
}

void HierarchyWidget::disconnectPreviousObject()
{
	// transformation tab related
	disconnectTransformTab();

	// puppet tab related
	disconnect(ui->pushButton_PuppetGo, SIGNAL(clicked()), 0, 0);
	disconnect(ui->pushButton_PuppetStop, SIGNAL(clicked()), 0, 0);

	// rendering tab related
	disconnect(ui->radioButton_Fill,  SIGNAL(toggled(bool)), 0, 0);
	disconnect(ui->radioButton_Line,  SIGNAL(toggled(bool)), 0, 0);
	disconnect(ui->radioButton_Point, SIGNAL(toggled(bool)), 0, 0);
	disconnect(ui->comboBox_SahderFiles, 0, 0, 0);
	disconnect(ui->checkBox_DiffuseMap, 0, 0, 0);
	disconnect(ui->checkBox_NormalMap, 0, 0, 0);

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

	// light tab related
	disconnect(ui->comboBox_LightType, 0, 0, 0);

	// rigid body tab related
	disconnect(ui->comboBox_RigidBodyMotionType, 0, 0, 0);
	disconnect(ui->doubleSpinBox_RigidBodyMass, 0, 0, 0);
	disconnect(ui->doubleSpinBox_RigidBodyGravityFactor, 0, 0, 0);
	disconnect(ui->doubleSpinBox_RigidBodyRestitution, 0, 0, 0);
	disconnect(ui->doubleSpinBox_RigidBodySizeX, 0, 0, 0);
	disconnect(ui->doubleSpinBox_RigidBodySizeY, 0, 0, 0);
	disconnect(ui->doubleSpinBox_RigidBodySizeZ, 0, 0, 0);
	disconnect(ui->doubleSpinBox_RigidBodyRadius, 0, 0, 0);
	disconnect(ui->doubleSpinBox_RigidBodyLinearVelocityX, 0, 0, 0);
	disconnect(ui->doubleSpinBox_RigidBodyLinearVelocityY, 0, 0, 0);
	disconnect(ui->doubleSpinBox_RigidBodyLinearVelocityZ, 0, 0, 0);
	disconnect(ui->doubleSpinBox_RigidBodyAngularVelocityX, 0, 0, 0);
	disconnect(ui->doubleSpinBox_RigidBodyAngularVelocityY, 0, 0, 0);
	disconnect(ui->doubleSpinBox_RigidBodyAngularVelocityZ, 0, 0, 0);
	disconnect(ui->doubleSpinBox_RigidBodyImpulseX, 0, 0, 0);
	disconnect(ui->doubleSpinBox_RigidBodyImpulseY, 0, 0, 0);
	disconnect(ui->doubleSpinBox_RigidBodyImpulseZ, 0, 0, 0);
	disconnect(ui->doubleSpinBox_RigidBodyImpulsePointX, 0, 0, 0);
	disconnect(ui->doubleSpinBox_RigidBodyImpulsePointY, 0, 0, 0);
	disconnect(ui->doubleSpinBox_RigidBodyImpulsePointZ, 0, 0, 0);
	disconnect(ui->doubleSpinBox_RigidBodyAngularImpulseX, 0, 0, 0);
	disconnect(ui->doubleSpinBox_RigidBodyAngularImpulseY, 0, 0, 0);
	disconnect(ui->doubleSpinBox_RigidBodyAngularImpulseZ, 0, 0, 0);
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

	// construct the popup menu
	QMenu* popMenu = new QMenu(ui->treeWidget);
	popMenu->addAction(m_deleteAction);
	QMenu* addMenu = popMenu->addMenu("Add");
	addMenu->addAction(m_addRigidBodyAction);

	// show the menu
	popMenu->exec(QCursor::pos());
}

void HierarchyWidget::deleteGameObject()
{
	// take the object from the map, and delete it
	if (!m_currentObject) return;
	m_scene->objectManager()->deleteObject(m_currentObject->objectName());
	updateObjectTree();
	m_currentObject = NULL;
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
			if(m_currentMaterials.size() == 0) return true;
			QColor col = QColorDialog::getColor(m_currentMaterials[0]->m_ambientColor, this);
			if(col.isValid()) 
			{
				// apply the color to the particle system and color picker both
				ui->graphicsView_AmbientColorPicker->setBackgroundBrush(QBrush(col, Qt::DiagCrossPattern));
				
				// change the material of the model
				foreach(Material* mat, m_currentMaterials)
				{
					mat->m_ambientColor = col;
				}

				emit materialChanged();
			}
			return true;
		}
		else if (obj == ui->graphicsView_DiffuseColorPicker)
		{
			if(m_currentMaterials.size() == 0) return true;
			QColor col = QColorDialog::getColor(m_currentMaterials[0]->m_diffuseColor, this);
			if(col.isValid()) 
			{
				// apply the color to the particle system and color picker both
				ui->graphicsView_DiffuseColorPicker->setBackgroundBrush(QBrush(col, Qt::DiagCrossPattern));
				
				// change the material of the model
				foreach(Material* mat, m_currentMaterials)
				{
					mat->m_diffuseColor = col;
				}

				emit materialChanged();
			}
			return true;
		}
		else if (obj == ui->graphicsView_SpecularColorPicker)
		{
			if(m_currentMaterials.size() == 0) return true;
			QColor col = QColorDialog::getColor(m_currentMaterials[0]->m_specularColor, this);
			if(col.isValid()) 
			{
				// apply the color to the particle system and color picker both
				ui->graphicsView_SpecularColorPicker->setBackgroundBrush(QBrush(col, Qt::DiagCrossPattern));
				
				// change the material of the model
				foreach(Material* mat, m_currentMaterials)
				{
					mat->m_specularColor = col;
				}

				emit materialChanged();
			}
			return true;
		}
		else if (obj == ui->graphicsView_EmissiveColorPicker)
		{
			if(m_currentMaterials.size() == 0) return true;
			QColor col = QColorDialog::getColor(m_currentMaterials[0]->m_emissiveColor, this);
			if(col.isValid()) 
			{
				// apply the color to the particle system and color picker both
				ui->graphicsView_EmissiveColorPicker->setBackgroundBrush(QBrush(col, Qt::DiagCrossPattern));
				
				// change the material of the model
				foreach(Material* mat, m_currentMaterials)
				{
					mat->m_emissiveColor = col;
				}

				emit materialChanged();
			}
			return true;
		}
		else if (obj == ui->graphicsView_NormalMapPicker)
		{
			if (!m_currentShadingTech) return true;
			ComponentPtr comp = m_currentObject->getComponent("Model");
			ModelPtr model = comp.dynamicCast<IModel>();
// 			QString fileName = QFileDialog::getOpenFileName(0, tr("Select a normal map texture"),
// 				QFileInfo(model->fileName()).absolutePath(),
// 				tr("Normal Map(*.*)"));
			QString fileName = QFileDialog::getOpenFileName(0, tr("Select a normal map texture"),
				"../Resource/Textures",
				tr("Normal Map(*.*)"));
			if (!fileName.isEmpty())
			{
				// apply the texture to the material and color picker both
				TexturePtr texture_normalMap = m_scene->textureManager()->getTexture(fileName);
				if(!texture_normalMap)
				{
					texture_normalMap = m_scene->textureManager()->addTexture(fileName, fileName, Texture::Texture2D, Texture::NormalMap);
				}
				m_currentMaterials[0]->addTexture(texture_normalMap);
				emit materialChanged();

				ui->graphicsView_NormalMapPicker->scene()->clear();
				QPixmap tex = texture_normalMap->generateQPixmap();
				QGraphicsPixmapItem* item = new QGraphicsPixmapItem(tex);
				ui->graphicsView_NormalMapPicker->scene()->addItem(item);
				ui->graphicsView_NormalMapPicker->fitInView(item);
			}

			return true;
		}
		else if (obj == ui->graphicsView_LightColorPicker)
		{
			if (!m_currentLight) return true;
			QColor col = QColorDialog::getColor(m_currentLight->color(), this);
			if(col.isValid()) 
			{
				// apply the color to the particle system and color picker both
				ui->graphicsView_LightColorPicker->setBackgroundBrush(QBrush(col, Qt::DiagCrossPattern));

				// change the color of the light
				m_currentLight->setColor(col);
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

void HierarchyWidget::searchShaders()
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
		
		// only add the right types of shaders
		ui->comboBox_SahderFiles->addItem(fileName);
	}
	
	// connect it when the loading is finished
	connect(ui->comboBox_SahderFiles, SIGNAL(currentTextChanged(const QString&)), this, SLOT(changeShader(const QString&)));
}

void HierarchyWidget::changeShader( const QString& shaderFile )
{
	if (!m_currentShadingTech || m_currentShadingTech->shaderFileName() == shaderFile) return;
	m_currentShadingTech->applyShader(shaderFile);

	// re assign the material properties
	assignMaterial();
}

void HierarchyWidget::assignMaterial()
{
	if (!m_currentShadingTech || !m_currentObject || m_currentMaterials.size() == 0) return;
	m_currentShadingTech->enable();
	m_currentShadingTech->setMaterial(m_currentMaterials[0]);
}

void HierarchyWidget::onRigidBodyRestitutionSliderChange( int value )
{
	ui->doubleSpinBox_RigidBodyRestitution->setValue(value/(double)100);
}

void HierarchyWidget::onRigidBodyRestitutionDoubleBoxChange( double value )
{
	ui->horizontalSlider_RigidBodyRestitution->setValue(value * 100);

	// change the restitution of the rigid body

}

void HierarchyWidget::onShininessSliderChange( int value )
{
	ui->doubleSpinBox_Shininess->setValue(value);
}

void HierarchyWidget::onShininessDoubleBoxChange( double value )
{
	ui->horizontalSlider_Shininess->setValue(value);

	// change the material of the model
	if(m_currentMaterials.size() == 0) return;
	foreach(Material* mat, m_currentMaterials)
	{
		mat->m_shininess = value;
	}
	
	emit materialChanged();
}

void HierarchyWidget::onShininessStrengthSliderChange( int value )
{
	ui->doubleSpinBox_ShininessStrength->setValue(value/(double)100);
}

void HierarchyWidget::onShininessStrengthDoubleBoxChange( double value )
{
	ui->horizontalSlider_ShininessStrength->setValue(value * 100);
	
	// change the material of the model
	if(m_currentMaterials.size() == 0) return;
	foreach(Material* mat, m_currentMaterials)
	{
		mat->m_shininessStrength = value;
	}

	emit materialChanged();
}

void HierarchyWidget::onRoughnessSliderChange( int value )
{
	ui->doubleSpinBox_Roughness->setValue(value/(double)100);
}

void HierarchyWidget::onRoughnessDoubleBoxChange( double value )
{
	ui->horizontalSlider_Roughness->setValue(value * 100);
	
	// change the material of the model
	if(m_currentMaterials.size() == 0) return;
	foreach(Material* mat, m_currentMaterials)
	{
		mat->m_roughness = value;
	}

	emit materialChanged();
}

void HierarchyWidget::onFresnelReflectanceSliderChange( int value )
{
	ui->doubleSpinBox_fresnelReflectance->setValue(value/(double)100);
}

void HierarchyWidget::onFresnelReflectanceDoubleBoxChange( double value )
{
	ui->horizontalSlider_fresnelReflectance->setValue(value * 100);

	// change the material of the model
	if(m_currentMaterials.size() == 0) return;
	foreach(Material* mat, m_currentMaterials)
	{
		mat->m_fresnelReflectance = value;
	}

	emit materialChanged();
}

void HierarchyWidget::onRefractiveIndexDoubleBoxChange( double value )
{
	// change the material of the model
	if(m_currentMaterials.size() == 0) return;
	foreach(Material* mat, m_currentMaterials)
	{
		mat->m_refractiveIndex = value;
	}

	emit materialChanged();
}

void HierarchyWidget::onConstantAttenuationSliderChange( int value )
{
	ui->doubleSpinBox_LightAttConst->setValue(value/(double)100);
}

void HierarchyWidget::onConstantAttenuationDoubleBoxChange( double value )
{
	ui->horizontalSlider_LightAttConst->setValue(value * 100);

	// change the light property
	if(m_currentLight) m_currentLight->setConstantAttenuation(value);
}

void HierarchyWidget::onLinearAttenuationSliderChange( int value )
{
	ui->doubleSpinBox_LightAttLinear->setValue(value/(double)100);
}

void HierarchyWidget::onLinearAttenuationDoubleBoxChange( double value )
{
	ui->horizontalSlider_LightAttLinear->setValue(value * 100);

	// change the light property
	if(m_currentLight) m_currentLight->setLinearAttenuation(value);
}

void HierarchyWidget::onQuadraticAttenuationSliderChange( int value )
{
	ui->doubleSpinBox_LightAttQuad->setValue(value/(double)100);
}

void HierarchyWidget::onQuadraticAttenuationDoubleBoxChange( double value )
{
	ui->horizontalSlider_LightAttQuad->setValue(value * 100);

	// change the light property
	if(m_currentLight) m_currentLight->setQuadraticAttenuation(value);
}

void HierarchyWidget::onLightIntensitySliderChange( int value )
{
	ui->doubleSpinBox_LightIntensity->setValue(value/(double)100);
}

void HierarchyWidget::onLightIntensityDoubleBoxChange( double value )
{
	ui->horizontalSlider_LightIntensity->setValue(value * 100);

	// change the light property
	if(m_currentLight) m_currentLight->setIntensity(value);
}

void HierarchyWidget::readShadingProperties()
{
	ui->graphicsView_DiffuseMapPicker->scene()->clear();
	ui->graphicsView_NormalMapPicker->scene()->clear();
	ui->checkBox_DiffuseMap->setChecked(false);
	ui->checkBox_NormalMap->setChecked(false);
	ComponentPtr comp = m_currentObject->getComponent("Model");
	ModelPtr model = comp.dynamicCast<IModel>();
	if (!model) return;
	searchShaders();

	m_currentShadingTech = model->renderingEffect().data();
	if (!m_currentShadingTech) return;

	ui->comboBox_SahderFiles->setCurrentText(m_currentShadingTech->shaderFileName());

	MaterialPtr mat = model->getMaterials()[0];
	// map the material information into the tab
	ui->graphicsView_AmbientColorPicker->setBackgroundBrush(QBrush(mat->m_ambientColor, Qt::DiagCrossPattern));
	ui->graphicsView_DiffuseColorPicker->setBackgroundBrush(QBrush(mat->m_diffuseColor, Qt::DiagCrossPattern));
	ui->graphicsView_SpecularColorPicker->setBackgroundBrush(QBrush(mat->m_specularColor, Qt::DiagCrossPattern));
	ui->graphicsView_EmissiveColorPicker->setBackgroundBrush(QBrush(mat->m_emissiveColor, Qt::DiagCrossPattern));

	ui->doubleSpinBox_Shininess->setValue(mat->m_shininess);
	ui->doubleSpinBox_ShininessStrength->setValue(mat->m_shininessStrength);
	ui->doubleSpinBox_Roughness->setValue(mat->m_roughness);
	ui->doubleSpinBox_fresnelReflectance->setValue(mat->m_fresnelReflectance);
	ui->doubleSpinBox_refractiveIndex->setValue(mat->m_refractiveIndex);
	// map the textures

	// normal map
	ui->graphicsView_DiffuseMapPicker->scene()->clear();
	ui->graphicsView_NormalMapPicker->scene()->clear();
	ui->graphicsView_SpecularMapPicker->scene()->clear();
	ui->graphicsView_EmissiveMapPicker->scene()->clear();

	if (mat->m_hasDiffuseMap)
	{
		ui->checkBox_DiffuseMap->setChecked(true);
		QPixmap tex = mat->getTexture(Texture::DiffuseMap)->generateQPixmap();
		QGraphicsPixmapItem* item = new QGraphicsPixmapItem(tex);
		ui->graphicsView_DiffuseMapPicker->scene()->addItem(item);
		ui->graphicsView_DiffuseMapPicker->fitInView(item);
	}
	if (mat->m_hasNormalMap)
	{
		ui->checkBox_NormalMap->setChecked(true);
		QPixmap tex = mat->getTexture(Texture::NormalMap)->generateQPixmap();
		QGraphicsPixmapItem* item = new QGraphicsPixmapItem(tex);
		ui->graphicsView_NormalMapPicker->scene()->addItem(item);
		ui->graphicsView_NormalMapPicker->fitInView(item);
	}
	
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

void HierarchyWidget::handleGameObjectTransformation( const vec3& pos, const vec3& rot, const vec3& scale )
{
	// disconnect the transform tab
	// because here its read only
	qDebug() << "current pos:" << pos;
}

void HierarchyWidget::disconnectTransformTab()
{
	disconnect(ui->doubleSpinBox_PositionX, SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_PositionY, SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_PositionZ, SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_RotationX, SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_RotationY, SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_RotationZ, SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_ScaleX,		SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_ScaleY,		SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_ScaleZ,		SIGNAL(valueChanged(double)), 0, 0);
	disconnect(ui->doubleSpinBox_ScaleFactor, 0, 0, 0);
	disconnect(ui->pushButton_Scale001, 0, 0, 0);
	disconnect(ui->pushButton_Scale01, 0, 0, 0);
	disconnect(ui->pushButton_Scale1, 0, 0, 0);
	disconnect(ui->pushButton_Scale10, 0, 0, 0);
	disconnect(ui->pushButton_Scale100, 0, 0, 0);
}

void HierarchyWidget::connectTransformTab()
{

	connect(ui->doubleSpinBox_PositionX, SIGNAL(valueChanged(double)), m_currentObject, SLOT(setFixedPositionX(double)));
	connect(ui->doubleSpinBox_PositionY, SIGNAL(valueChanged(double)), m_currentObject, SLOT(setFixedPositionY(double)));
	connect(ui->doubleSpinBox_PositionZ, SIGNAL(valueChanged(double)), m_currentObject, SLOT(setFixedPositionZ(double)));
	connect(ui->doubleSpinBox_RotationX, SIGNAL(valueChanged(double)), m_currentObject, SLOT(setFixedRotationX(double)));
	connect(ui->doubleSpinBox_RotationY, SIGNAL(valueChanged(double)), m_currentObject, SLOT(setFixedRotationY(double)));
	connect(ui->doubleSpinBox_RotationZ, SIGNAL(valueChanged(double)), m_currentObject, SLOT(setFixedRotationZ(double)));
	connect(ui->doubleSpinBox_ScaleX,	 SIGNAL(valueChanged(double)), m_currentObject, SLOT(setFixedScaleX(double)));
	connect(ui->doubleSpinBox_ScaleY,	 SIGNAL(valueChanged(double)), m_currentObject, SLOT(setFixedScaleY(double)));
	connect(ui->doubleSpinBox_ScaleZ,	 SIGNAL(valueChanged(double)), m_currentObject, SLOT(setFixedScaleZ(double)));
	connect(ui->doubleSpinBox_RotationX, SIGNAL(valueChanged(double)), this, SLOT(onRotationXSpinChange(double)));
	connect(ui->doubleSpinBox_RotationY, SIGNAL(valueChanged(double)), this, SLOT(onRotationYSpinChange(double)));
	connect(ui->doubleSpinBox_RotationZ, SIGNAL(valueChanged(double)), this, SLOT(onRotationZSpinChange(double)));
	connect(ui->doubleSpinBox_ScaleFactor, SIGNAL(valueChanged(double)), this, SLOT(onScaleFactorDoubleBoxChange(double)));
	connect(ui->pushButton_Scale001, SIGNAL(clicked()), this, SLOT(onScale001Pushed()));
	connect(ui->pushButton_Scale01, SIGNAL(clicked()), this, SLOT(onScale01Pushed()));
	connect(ui->pushButton_Scale1, SIGNAL(clicked()), this, SLOT(onScale1Pushed()));
	connect(ui->pushButton_Scale10, SIGNAL(clicked()), this, SLOT(onScale10Pushed()));
	connect(ui->pushButton_Scale100, SIGNAL(clicked()), this, SLOT(onScale100Pushed()));

}

void HierarchyWidget::fillInTransformTab()
{
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
}

void HierarchyWidget::onObjectPicked( GameObjectPtr selected )
{
	// make the camera follow the object
	//m_scene->getCamera()->followTarget(m_currentObject);

	// if nothing is selected
	if (!selected)
	{
		ui->treeWidget->setCurrentIndex(ui->treeWidget->rootIndex());
		ui->tabWidget->removeTab(ui->tabWidget->indexOf(m_renderingTab));
		ui->tabWidget->removeTab(ui->tabWidget->indexOf(m_particleSystemTab));
		disconnectPreviousObject();
		clearReference();
		return;
	}

	// find the item that has this name
	//QTreeWidgetItem* selected = ui->treeWidget->findItems(name, Qt::MatchExactly).first();
	QList<QTreeWidgetItem*> items = ui->treeWidget->findItems(selected->objectName(), Qt::MatchRecursive);

	// this usually won't happen
	if (items.size() == 0) return;

	ui->treeWidget->setCurrentItem(items.first());
}

void HierarchyWidget::readLightSourceProperties(LightPtr light)
{
	m_currentLight = light.data();
	Light::LightType type = light->type();

	if (type == Light::PointLight)
	{
		ui->comboBox_LightType->setCurrentText("Point Light");
		// show the attenuation group
		ui->groupBox_LightAtt->show();
	}
	else if (type == Light::DirectionalLight)
	{
		ui->comboBox_LightType->setCurrentText("Directional Light");
		// hide the attenuation group
		ui->groupBox_LightAtt->hide();
	}
	else if (type == Light::SpotLight)
	{
		ui->comboBox_LightType->setCurrentText("Spot Light");
		// show the attenuation group
		ui->groupBox_LightAtt->show();
	}
	else if (type == Light::AmbientLight)
		ui->comboBox_LightType->setCurrentText("Ambient Light");
	else if (type == Light::AreaLight)
		ui->comboBox_LightType->setCurrentText("Area Light");

	ui->graphicsView_LightColorPicker->setBackgroundBrush(QBrush(light->color(), Qt::DiagCrossPattern));

	ui->doubleSpinBox_LightAttConst->setValue(light->constantAttenuation());
	ui->doubleSpinBox_LightAttLinear->setValue(light->linearAttenuation());
	ui->doubleSpinBox_LightAttQuad->setValue(light->quadraticAttenuation());

	// connect the combo box of light type
	connect(ui->comboBox_LightType, SIGNAL(currentTextChanged(const QString&)), this, SLOT(changeLightType(const QString&)));
}

void HierarchyWidget::changeLightType( const QString& type )
{
	if(!m_currentLight) return;

	if (type == "Point Light")
	{
		m_currentLight->setType(Light::PointLight);
		// show the attenuation group
		ui->groupBox_LightAtt->show();
	}
	else if (type == "Directional Light")
	{
		m_currentLight->setType(Light::DirectionalLight);
		// hide the attenuation group
		ui->groupBox_LightAtt->hide();
	}
	else if (type == "Spot Light")
	{
		m_currentLight->setType(Light::SpotLight);
		// show the attenuation group
		ui->groupBox_LightAtt->show();
	}
	else if (type == "Ambient Light")
		m_currentLight->setType(Light::AmbientLight);
	else if (type == "Area Light")
		m_currentLight->setType(Light::AreaLight);
}	

void HierarchyWidget::onScaleFactorDoubleBoxChange( double value )
{
	if (!m_currentObject)
	{
		return;
	}
	m_currentObject->setScale(value);
	disconnect(ui->doubleSpinBox_ScaleX, 0, 0, 0);
	disconnect(ui->doubleSpinBox_ScaleY, 0, 0, 0);
	disconnect(ui->doubleSpinBox_ScaleZ, 0, 0, 0);
	ui->doubleSpinBox_ScaleX->setValue(value);
	ui->doubleSpinBox_ScaleY->setValue(value);
	ui->doubleSpinBox_ScaleZ->setValue(value);
	connect(ui->doubleSpinBox_ScaleX,	 SIGNAL(valueChanged(double)), m_currentObject, SLOT(setFixedScaleX(double)));
	connect(ui->doubleSpinBox_ScaleY,	 SIGNAL(valueChanged(double)), m_currentObject, SLOT(setFixedScaleX(double)));
	connect(ui->doubleSpinBox_ScaleZ,	 SIGNAL(valueChanged(double)), m_currentObject, SLOT(setFixedScaleX(double)));
}

void HierarchyWidget::onScale001Pushed()
{
	ui->doubleSpinBox_ScaleFactor->setValue(0.01);
}

void HierarchyWidget::onScale01Pushed()
{
	ui->doubleSpinBox_ScaleFactor->setValue(0.1);
}

void HierarchyWidget::onScale1Pushed()
{
	ui->doubleSpinBox_ScaleFactor->setValue(1);
}

void HierarchyWidget::onScale10Pushed()
{
	ui->doubleSpinBox_ScaleFactor->setValue(10);
}

void HierarchyWidget::onScale100Pushed()
{
	ui->doubleSpinBox_ScaleFactor->setValue(100);
}

void HierarchyWidget::readRigidBodyProperties( RigidBodyPtr rb )
{
	RigidBody::MotionType motionType = rb->getMotionType();
	vec3 linearVelocity =  rb->getLinearVelocity();
	vec3 angularVelocity = rb->getAngularVelocity();
	SphereColliderPtr sphere = rb->getBroadPhaseCollider().dynamicCast<SphereCollider>();
	BoxColliderPtr box = rb->getBroadPhaseCollider().dynamicCast<BoxCollider>();

	switch(motionType)
	{
	case RigidBody::MOTION_BOX_INERTIA:
		ui->comboBox_RigidBodyMotionType->setCurrentText("Box");
		ui->doubleSpinBox_RigidBodySizeX->setValue(box->getHalfExtents().x() * 2);
		ui->doubleSpinBox_RigidBodySizeY->setValue(box->getHalfExtents().y() * 2);
		ui->doubleSpinBox_RigidBodySizeZ->setValue(box->getHalfExtents().z() * 2);
		break;

	case RigidBody::MOTION_SPHERE_INERTIA:
		ui->comboBox_RigidBodyMotionType->setCurrentText("Sphere");
		ui->doubleSpinBox_RigidBodyRadius->setValue(sphere->getRadius());
		break;

	case RigidBody::MOTION_FIXED:
		ui->comboBox_RigidBodyMotionType->setCurrentText("Fixed");
		ui->doubleSpinBox_RigidBodySizeX->setValue(box->getHalfExtents().x() * 2);
		ui->doubleSpinBox_RigidBodySizeY->setValue(box->getHalfExtents().y() * 2);
		ui->doubleSpinBox_RigidBodySizeZ->setValue(box->getHalfExtents().z() * 2);
		break;
	}

	ui->doubleSpinBox_RigidBodyMass->setValue(rb->getMass());
	ui->doubleSpinBox_RigidBodyGravityFactor->setValue(rb->getGravityFactor());
	ui->doubleSpinBox_RigidBodyRestitution->setValue(rb->getRestitution());
	ui->horizontalSlider_RigidBodyRestitution->setValue(rb->getRestitution() * 100);
	ui->doubleSpinBox_RigidBodyLinearVelocityX->setValue(linearVelocity.x());
	ui->doubleSpinBox_RigidBodyLinearVelocityY->setValue(linearVelocity.y());
	ui->doubleSpinBox_RigidBodyLinearVelocityZ->setValue(linearVelocity.z());
	ui->doubleSpinBox_RigidBodyAngularVelocityX->setValue(angularVelocity.x());
	ui->doubleSpinBox_RigidBodyAngularVelocityY->setValue(angularVelocity.y());
	ui->doubleSpinBox_RigidBodyAngularVelocityZ->setValue(angularVelocity.z());
}

void HierarchyWidget::toggleDiffuseMap( bool state )
{
	// enable or disable the diffuse map
	foreach(Material* mat, m_currentMaterials)
	{
		mat->m_hasDiffuseMap = state;
	}

	emit materialChanged();
}

void HierarchyWidget::toggleNormalMap( bool state )
{
	// enable or disable the diffuse map
	foreach(Material* mat, m_currentMaterials)
	{
		mat->m_hasNormalMap = state;
	}

	emit materialChanged();
}

void HierarchyWidget::clearReference()
{
	m_currentObject = NULL;
	m_currentShadingTech = NULL;
	m_currentLight = NULL;
	m_currentMaterials.clear();
}

void HierarchyWidget::createRigidBody()
{
	if (!m_currentObject) return;

	m_scene->createRigidBody(m_currentObject);
	readCurrentGameObject();
}

void HierarchyWidget::connectRigidBodyTab( RigidBodyPtr rb )
{
	connect(ui->comboBox_RigidBodyMotionType, SIGNAL(currentTextChanged(const QString&)), rb.data(), SLOT(setMotionType_SLOT(const QString&)));
	connect(ui->doubleSpinBox_RigidBodyGravityFactor, SIGNAL(valueChanged(double)), rb.data(), SLOT(setGravityFactor_SLOT(double)));
	connect(ui->doubleSpinBox_RigidBodyRestitution, SIGNAL(valueChanged(double)), rb.data(), SLOT(setRestitution_SLOT(double)));
	connect(ui->doubleSpinBox_RigidBodyMass, SIGNAL(valueChanged(double)), rb.data(), SLOT(setMass_SLOT(double)));
	connect(ui->doubleSpinBox_RigidBodySizeX, SIGNAL(valueChanged(double)), rb.data(), SLOT(setExtentsX_SLOT(double)));
	connect(ui->doubleSpinBox_RigidBodySizeY, SIGNAL(valueChanged(double)), rb.data(), SLOT(setExtentsY_SLOT(double)));
	connect(ui->doubleSpinBox_RigidBodySizeZ, SIGNAL(valueChanged(double)), rb.data(), SLOT(setExtentsZ_SLOT(double)));
	connect(ui->doubleSpinBox_RigidBodyRadius, SIGNAL(valueChanged(double)), rb.data(), SLOT(setRadius_SLOT(double)));
	connect(ui->doubleSpinBox_RigidBodyLinearVelocityX,  SIGNAL(valueChanged(double)), rb.data(), SLOT(setLinearVelocityX_SLOT(double)));
	connect(ui->doubleSpinBox_RigidBodyLinearVelocityY,  SIGNAL(valueChanged(double)), rb.data(), SLOT(setLinearVelocityY_SLOT(double)));
	connect(ui->doubleSpinBox_RigidBodyLinearVelocityZ,  SIGNAL(valueChanged(double)), rb.data(), SLOT(setLinearVelocityZ_SLOT(double)));
	connect(ui->doubleSpinBox_RigidBodyAngularVelocityX, SIGNAL(valueChanged(double)), rb.data(), SLOT(setAngularVelocityX_SLOT(double)));
	connect(ui->doubleSpinBox_RigidBodyAngularVelocityY, SIGNAL(valueChanged(double)), rb.data(), SLOT(setAngularVelocityY_SLOT(double)));
	connect(ui->doubleSpinBox_RigidBodyAngularVelocityZ, SIGNAL(valueChanged(double)), rb.data(), SLOT(setAngularVelocityZ_SLOT(double)));
	connect(ui->doubleSpinBox_RigidBodyImpulseX, SIGNAL(valueChanged(double)), rb.data(), SLOT(setPointImpulseX_SLOT(double)));
	connect(ui->doubleSpinBox_RigidBodyImpulseY, SIGNAL(valueChanged(double)), rb.data(), SLOT(setPointImpulseY_SLOT(double)));
	connect(ui->doubleSpinBox_RigidBodyImpulseZ, SIGNAL(valueChanged(double)), rb.data(), SLOT(setPointImpulseZ_SLOT(double)));
	connect(ui->doubleSpinBox_RigidBodyImpulsePointX, SIGNAL(valueChanged(double)), rb.data(), SLOT(setPointImpulsePositionX_SLOT(double)));
	connect(ui->doubleSpinBox_RigidBodyImpulsePointY, SIGNAL(valueChanged(double)), rb.data(), SLOT(setPointImpulsePositionY_SLOT(double)));
	connect(ui->doubleSpinBox_RigidBodyImpulsePointZ, SIGNAL(valueChanged(double)), rb.data(), SLOT(setPointImpulsePositionZ_SLOT(double)));
	connect(ui->doubleSpinBox_RigidBodyAngularImpulseX, SIGNAL(valueChanged(double)), rb.data(), SLOT(setAngularImpulseX_SLOT(double)));
	connect(ui->doubleSpinBox_RigidBodyAngularImpulseY, SIGNAL(valueChanged(double)), rb.data(), SLOT(setAngularImpulseY_SLOT(double)));
	connect(ui->doubleSpinBox_RigidBodyAngularImpulseZ, SIGNAL(valueChanged(double)), rb.data(), SLOT(setAngularImpulseZ_SLOT(double)));
}
