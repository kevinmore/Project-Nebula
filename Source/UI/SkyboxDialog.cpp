#include "SkyboxDialog.h"
#include "ui_SkyboxWidget.h"
#include <QGraphicsPixmapItem>

SkyboxDialog::SkyboxDialog( SkyboxPtr skybox, QWidget *parent /*= 0*/ )
	: QDialog(parent),
	  m_skybox(skybox),
	  ui(new Ui::Dialog)
{
	ui->setupUi(this);
	connect(ui->buttonBox, SIGNAL(accepted()), this, SLOT(setSkyboxTextures()));
	connect(ui->buttonBox, SIGNAL(rejected()), this, SLOT(close()));

	m_views << ui->graphicsView_back << ui->graphicsView_front << ui->graphicsView_left
		    << ui->graphicsView_right << ui->graphicsView_top << ui->graphicsView_bottom;

	foreach(QGraphicsView* view, m_views)
	{
		view->setScene(new QGraphicsScene(this));
	}

	unfoldSkybox();
}


SkyboxDialog::~SkyboxDialog()
{
	delete ui;
}

void SkyboxDialog::setSkyboxTextures()
{
	qDebug() << "Accepted";
	close();
}

void SkyboxDialog::unfoldSkybox()
{
	QVector<QPixmap> textures = m_skybox->getCubemapTexture()->getQPixmaps();

	for(int i = 0; i < textures.size(); ++i)
	{
		QGraphicsPixmapItem* item = new QGraphicsPixmapItem(textures[i]);
		m_views[i]->scene()->clear();
		m_views[i]->scene()->addItem(item);
		m_views[i]->fitInView(item);
	}
}
