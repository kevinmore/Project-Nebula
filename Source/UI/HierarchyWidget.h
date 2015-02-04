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

	QAction* m_deleteAction;
	QWidget* particleSystemTab;

	void readHierarchy(GameObject* go, QTreeWidgetItem* parentItem); // go through the game objects
	void resetHierarchy(GameObject* go); // reset every game object from the given one
	void clearTransformationArea();
	void readShadingProperties();
	void connectParticleSystemTab(ParticleSystemPtr ps);
	void readParticleSystemConfig(ParticleSystemPtr ps);
	void searchSuitableShaders(ModelPtr currentModel);
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
	void deleteGameObject();
	void handleGameObjectTransformation(const vec3& pos, const vec3& rot, const vec3& scale);
	void showMouseRightButton(const QPoint& point);
	void setColorPickerEnabled(bool status);
	void changeShader(const QString& shaderFile);
	void onShininessSliderChange(int value);
	void onShininessDoubleBoxChange(double value);
	void onShininessStrengthSliderChange(int value);
	void onShininessStrengthDoubleBoxChange(double value);

	void onRotationXDialChange(int val);
	void onRotationYDialChange(int val);
	void onRotationZDialChange(int val);
	void onRotationXSpinChange(double val);
	void onRotationYSpinChange(double val);
	void onRotationZSpinChange(double val);

	void assignPuppet();

public slots:
	void updateObjectTree();
	void onObjectPicked(const QString& name);

protected:
	bool eventFilter(QObject *obj, QEvent *ev); // install a filter event for the color picker
};

