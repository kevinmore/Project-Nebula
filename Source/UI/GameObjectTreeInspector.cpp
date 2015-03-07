#include "GameObjectTreeInspector.h"
#include "HierarchyWidget.h"
#include <Scene/Scene.h>

GameObjectTreeInspector::GameObjectTreeInspector( QWidget *parent /*= 0*/ )
	: QTreeWidget(parent),
	  m_scene(Scene::instance())
{}

void GameObjectTreeInspector::setContainerWidget( HierarchyWidget* widget )
{
	m_container = widget;
}

void GameObjectTreeInspector::dropEvent( QDropEvent * event )
{
	// retrieve the source and destination game objects
	QTreeWidgetItem* item = itemAt(event->pos());
	if (!item) return;

	GameObject* dest = ObjectManager::instance()->getGameObject(item->text(0)).data();
	GameObject* source = m_container->getCurrentGameObject();

	// validate
	if (!source || !dest || source == dest)
	{
		if (itemAt(event->pos()) == topLevelItem(0))
		{
			source->setParent(m_scene->sceneRoot());
			finishEvent(event);
		}
		return;
	}

	// change the parent of the current game object
	source->setParent(dest);
	finishEvent(event);
}

void GameObjectTreeInspector::finishEvent( QDropEvent * event )
{
	emit m_scene->updateHierarchy();
	QTreeWidget::dropEvent(event);
}

