#pragma once
#include <QObject>
class LogCenter : public QObject
{
	Q_OBJECT

public:
	static LogCenter* instance();

signals:
	void message(QtMsgType type, const QMessageLogContext &context, const QString &msg);

private:
	LogCenter();
	~LogCenter();
	static LogCenter* m_instance;
};

