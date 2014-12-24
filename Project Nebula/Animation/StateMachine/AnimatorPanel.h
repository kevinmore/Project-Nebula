#pragma once
#include <QtWidgets>

class AnimatorPanel : public QWidget
{
public:
	AnimatorPanel();
	~AnimatorPanel();

	// buttons
	QPushButton *interactButton, *moveButton, *fastMoveButton, *turnLeftButton, *turnRightButton, *turnRoundButton;
};

