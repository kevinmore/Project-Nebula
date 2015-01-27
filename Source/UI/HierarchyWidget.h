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
	QAction* m_deleteAction;
	QWidget* particleSystemTab;

	void readHierarchy(GameObject* go, QTreeWidgetItem* parentItem); // go through the game objects
	void resetHierarchy(GameObject* go); // reset every game object from the given one
	void clearTransformationArea();
	void connectParticleSystemTab(ParticleSystemPtr ps);
	void readParticleSystemConfig(ParticleSystemPtr ps);
	void searchShaders();

private slots:
	void connectCurrentObject();
	void disconnectPreviousObject();
	void readGameObject(QTreeWidgetItem* current, QTreeWidgetItem* previous);
	void resetSelectedObject();
	void renameGameObject(QTreeWidgetItem * item, int column);
	void deleteGameObject();
	void showMouseRightButton(const QPoint& point);
	void setColorPickerEnabled(bool status);
	void shaderComboboxChanged(const QString& shaderFile);

public slots:
	void updateObjectTree();

public: signals:
	void shaderSelected(GameObject* target, const QString& shaderFile);

protected:
	bool eventFilter(QObject *obj, QEvent *ev); // install a filter event for the color picker
};

