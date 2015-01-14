#pragma once
#include <QObject>
class LogCenter : public QObject
{
	Q_OBJECT

public:
	static LogCenter* instance();
	bool m_wirteToFile;

signals:
	void message(QtMsgType type, const QMessageLogContext &context, const QString &msg);

public slots:
	void toggleWriteToFile(bool state);

private:
	LogCenter();
	~LogCenter();
	static LogCenter* m_instance;
};

