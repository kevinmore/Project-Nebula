#pragma once
#include <QTreeWidget>

class HierarchyWidget;
class Scene;
class GameObjectTreeInspector : public QTreeWidget
{

public:
	GameObjectTreeInspector(QWidget *parent = 0);
	~GameObjectTreeInspector(){}
	void setContainerWidget(HierarchyWidget* widget);

protected:
	void dropEvent(QDropEvent * event);
	void finishEvent(QDropEvent * event);
private:
	Scene* m_scene;
	HierarchyWidget* m_container;
};
