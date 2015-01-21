#include "SkyboxDialog.h"
#include "ui_SkyboxWidget.h"

SkyboxDialog::SkyboxDialog( Skybox* skybox, QWidget *parent /*= 0*/ )
	: QDialog(parent),
	  m_skybox(skybox),
	  ui(new Ui::Dialog)
{
	ui->setupUi(this);
	connect(ui->buttonBox, SIGNAL(accepted()), this, SLOT(setSkyboxTextures()));
	connect(ui->buttonBox, SIGNAL(rejected()), this, SLOT(close()));
}


SkyboxDialog::~SkyboxDialog()
{
	delete ui;
}

void SkyboxDialog::setSkyboxTextures()
{
	qDebug() << "Accepted";
}
