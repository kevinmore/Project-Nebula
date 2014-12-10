#include "AnimatorPanel.h"

AnimatorPanel::AnimatorPanel()
{
	setWindowTitle("Animator Control Panel");
	resize(400, 300);

	interactButton = new QPushButton(this);
	interactButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	interactButton->setText("Interact");

	moveButton = new QPushButton(this);
	moveButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

	fastMoveButton = new QPushButton(this);
	fastMoveButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

	turnLeftButton = new QPushButton(this);
	turnLeftButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	turnLeftButton->setText("Turn Left");

	turnRightButton = new QPushButton(this);
	turnRightButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	turnRightButton->setText("Turn Right");

	turnRoundButton = new QPushButton(this);
	turnRoundButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	turnRoundButton->setText("Turn Around");


	QGridLayout* mainLayout = new QGridLayout;
	mainLayout->addWidget(moveButton, 0, 0);
	mainLayout->addWidget(fastMoveButton, 0, 2);

	mainLayout->addWidget(turnLeftButton, 1, 0);
	mainLayout->addWidget(interactButton, 1, 1);
	mainLayout->addWidget(turnRightButton, 1, 2);

	mainLayout->addWidget(turnRoundButton, 2, 1);
	setLayout(mainLayout);
}


AnimatorPanel::~AnimatorPanel(void)
{
}
