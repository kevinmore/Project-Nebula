#pragma once
#include <QWidget>
#include <Scene/Scene.h>
#include <UI/Canvas.h>

namespace Ui {
	class HierarchyViewer;
}

class HierarchyWidget : public QWidget
{
	Q_OBJECT

public:
	HierarchyWidget(Scene* scene, Canvas* canvas, QWidget *parent = 0);
	~HierarchyWidget();

private:
	Ui::HierarchyViewer *ui;
	Canvas* m_canvas;
	Scene* m_scene;
	GameObject* m_currentObject;
	ShadingTechnique* m_currentShadingTech;
	QVector<Material*> m_currentMaterials;
	Light* m_currentLight;

	QAction* m_deleteAction;
	QWidget *m_renderingTab, *m_particleSystemTab, *m_lightTab, *m_rigidBodyTab;

	void readHierarchy(GameObject* go, QTreeWidgetItem* parentItem); // go through the game objects
	void resetHierarchy(GameObject* go); // reset every game object from the given one
	void clearTransformationArea();
	void readShadingProperties();
	void readLightSourceProperties(LightPtr light);
	void connectParticleSystemTab(ParticleSystemPtr ps);
	void readParticleSystemConfig(ParticleSystemPtr ps);
	void readRigidBodyProperties(RigidBodyPtr rb);
	void searchShaders();
	void connectSliderBarAndDoubleSpinBox(QSlider* slider, QDoubleSpinBox* box);

	void disconnectTransformTab();
	void connectTransformTab();
	void fillInTransformTab();

private slots:
	void connectCurrentObject();
	void disconnectPreviousObject();
	void readGameObject(QTreeWidgetItem* current, QTreeWidgetItem* previous);
	void resetSelectedObject();
	void renameGameObject(QTreeWidgetItem * item, int column);
	void handleGameObjectTransformation(const vec3& pos, const vec3& rot, const vec3& scale);
	void showMouseRightButton(const QPoint& point);
	void setColorPickerEnabled(bool status);
	void changeShader(const QString& shaderFile);
	void changeLightType(const QString& type);

	void onShininessSliderChange(int value);
	void onShininessDoubleBoxChange(double value);

	void onShininessStrengthSliderChange(int value);
	void onShininessStrengthDoubleBoxChange(double value);

	void onRoughnessSliderChange(int value);
	void onRoughnessDoubleBoxChange(double value);

	void onFresnelReflectanceSliderChange(int value);
	void onFresnelReflectanceDoubleBoxChange(double value);

	void onRefractiveIndexDoubleBoxChange(double value);

	void onConstantAttenuationSliderChange(int value);
	void onConstantAttenuationDoubleBoxChange(double value);

	void onLinearAttenuationSliderChange(int value);
	void onLinearAttenuationDoubleBoxChange(double value);

	void onQuadraticAttenuationSliderChange(int value);
	void onQuadraticAttenuationDoubleBoxChange(double value);

	void onLightIntensitySliderChange(int value);
	void onLightIntensityDoubleBoxChange(double value);

	void onRotationXDialChange(int val);
	void onRotationYDialChange(int val);
	void onRotationZDialChange(int val);
	void onRotationXSpinChange(double val);
	void onRotationYSpinChange(double val);
	void onRotationZSpinChange(double val);

	void onScaleFactorDoubleBoxChange(double value);
	void onScale001Pushed();
	void onScale01Pushed();
	void onScale1Pushed();
	void onScale10Pushed();
	void onScale100Pushed();
	void assignPuppet();


public slots:
	void deleteGameObject();
	void updateObjectTree();
	void onObjectPicked(GameObjectPtr selected);
	void assignMaterial();

signals:
	void materialChanged();

protected:
	bool eventFilter(QObject *obj, QEvent *ev); // install a filter event for the color picker
};

