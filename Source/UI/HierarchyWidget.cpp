#include "HierarchyWidget.h"
#include "ui_HierarchyWidget.h"

HierarchyWidget::HierarchyWidget(Scene* scene, QWidget *parent)
	: QWidget(parent),
	  m_scene(scene),
	  ui(new Ui::HierarchyViewer)

{
	ui->setupUi(this);
	ui->treeWidget->setHeaderLabel("Game Objects");
	connect(m_scene, SIGNAL(updateHierarchy()), this, SLOT(resetTree()));
	resetTree();
}

HierarchyWidget::~HierarchyWidget()
{
	delete ui;
}

void HierarchyWidget::resetTree()
{
	qDebug() << "reseting!";
	ui->treeWidget->clear();

	QTreeWidgetItem* rootItem = new QTreeWidgetItem(ui->treeWidget);
	rootItem->setText(0, m_scene->sceneNode()->objectName());

	QObjectList childrenList = m_scene->sceneNode()->children();
	qDebug() << "size:" << childrenList.count();
	foreach(QObject* obj, childrenList)
	{
		QTreeWidgetItem* item = new QTreeWidgetItem(rootItem);
		item->setText(0, obj->objectName());
	}

	ui->treeWidget->expandAll();
}
