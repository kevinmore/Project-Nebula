#include "GameObjectTreeInspector.h"
#include "HierarchyWidget.h"
#include <Scene/Scene.h>

GameObjectTreeInspector::GameObjectTreeInspector( QWidget *parent /*= 0*/ )
	: QTreeWidget(parent)
{

}

void GameObjectTreeInspector::setScene( Scene* scene )
{
	m_scene = scene;
}

void GameObjectTreeInspector::setContainerWidget( HierarchyWidget* widget )
{
	m_container = widget;
}

void GameObjectTreeInspector::dropEvent( QDropEvent * event )
{
	// retrieve the source and destination game objects
	GameObject* dest = m_scene->objectManager()->getGameObject(itemAt(event->pos())->text(0)).data();
	GameObject* source = m_container->getCurrentGameObject();

	// validate
	if (!source || !dest || source == dest)
	{
		if (itemAt(event->pos()) == topLevelItem(0))
		{
			source->setParent(m_scene->sceneRoot());
			QTreeWidget::dropEvent(event);
		}
		return;
	}

	// change the parent of the current game object
	source->setParent(dest);

	QTreeWidget::dropEvent(event);
}

