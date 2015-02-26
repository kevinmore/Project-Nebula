#pragma once
#include <QTreeWidget>

class HierarchyWidget;
class Scene;
class GameObjectTreeInspector : public QTreeWidget
{

public:
	GameObjectTreeInspector(QWidget *parent = 0);
	~GameObjectTreeInspector(){}
	void setScene(Scene* scene);
	void setContainerWidget(HierarchyWidget* widget);

protected:
	void dropEvent(QDropEvent * event);

private:
	Scene* m_scene;
	HierarchyWidget* m_container;
};
