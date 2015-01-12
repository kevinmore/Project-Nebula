#pragma once
#include <QWidget>
#include <Scene/Scene.h>

namespace Ui {
	class HierarchyViewer;
}

class HierarchyWidget : public QWidget
{
	Q_OBJECT

public:
	HierarchyWidget(Scene* scene, QWidget *parent = 0);
	~HierarchyWidget();

private:
	Ui::HierarchyViewer *ui;
	Scene* m_scene;
	GameObject* m_currentObject;

	void readHierarchy(GameObject* go, QTreeWidgetItem* parentItem); // go through the game objects

private slots:
	void connectCurrentObject();
	void disconnectPreviousObject();
	void clearTransformationArea();
	void updateTransformation(QTreeWidgetItem* current, QTreeWidgetItem* previous);

public slots:
	void updateObjectTree();
};

