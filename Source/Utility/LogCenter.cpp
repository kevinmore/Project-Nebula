#include "LogCenter.h"
#include <QMetaType>
#include <QMutex>
#include <QMutexLocker>
#include <QCoreApplication>
#include <QtGlobal>
#include <QDateTime>
#include <QLoggingCategory>
#include <QFile>

void static logCenterFunction(QtMsgType type, const QMessageLogContext &context, const QString &msg)
{
	// detail the information
	QString dt = QDateTime::currentDateTime().toString(Qt::ISODate);
	QString strLog = QString("\n[%1] ").arg(dt);

	switch (type)
	{
	case QtDebugMsg:
		strLog += QString("<Message>\n%1\n").arg(msg);
		break;
	case QtWarningMsg:
		strLog += QString("<Warning>\n%1\n").arg(msg);
		break;
	case QtCriticalMsg:
		strLog += QString("<Critical>\n%1\n").arg(msg);
		break;
	case QtFatalMsg:
		strLog += QString("<Fatal>\n%1\n").arg(msg);
		abort();
		break;
	}

	// delegate
	QMetaObject::invokeMethod(LogCenter::instance(), "message"
		, Q_ARG(QtMsgType, type)
		, Q_ARG(QMessageLogContext, context)
		, Q_ARG(QString, strLog));

	// we still want to print it into the console
	fprintf(stderr, "%s", strLog.toLocal8Bit().constData());

	// save the outputs into a log file
	if(!LogCenter::instance()->m_wirteToFile) return;
	QFile logFile("./Log.log");

	if (!logFile.open(QIODevice::WriteOnly | QIODevice::Append))
	{
		return;
	}

	QTextStream ts(&logFile);
	ts << strLog << endl;

	logFile.flush();
	logFile.close();
}

LogCenter * LogCenter::m_instance = 0;

LogCenter * LogCenter::instance()
{
	static QMutex mutex;
	if (!m_instance) {
		QMutexLocker locker(&mutex);
		if (!m_instance)
			m_instance = new LogCenter;
	}

	return m_instance;
}

LogCenter::LogCenter()
	: QObject(qApp),
	  m_wirteToFile(false)
{
	qRegisterMetaType<QtMsgType>("QtMsgType");
	qInstallMessageHandler(logCenterFunction);
}


LogCenter::~LogCenter(void)
{
}

void LogCenter::toggleWriteToFile( bool state )
{
	m_wirteToFile = state;
}

