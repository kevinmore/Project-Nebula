/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef PLATFORM_INIT_TIZEN_H
#define PLATFORM_INIT_TIZEN_H


#include <FApp.h>
#include <FBase.h>
#include <FSystem.h>
#include <FUi.h>
#include <FGrpGlPlayer.h>

#include <Common/Base/hkBase.h>

class TizenForm;

class AppThread : public Tizen::Base::Runtime::Thread
{
public:
	AppThread() {};
	virtual ~AppThread() {};
	result Construct()
	{
			return Tizen::Base::Runtime::Thread::Construct();
	}
	Object* Run();
 };
 
class TizenApp :
	public Tizen::App::Application, 
	public Tizen::System::IScreenEventListener,
	public Tizen::Base::Runtime::ITimerEventListener
{
public:

	static Tizen::App::Application* CreateInstance();

	TizenApp();
	virtual ~TizenApp();

	virtual TizenForm* createForm();

	virtual bool OnAppInitializing(Tizen::App::AppRegistry& appRegistry);
	virtual bool OnAppTerminating(Tizen::App::AppRegistry& appRegistry, bool forcedTermination = false);

	// Called when the UiApp initializing is finished. 
	virtual bool OnAppInitialized(); 

	virtual void OnForeground();
	virtual void OnBackground();
	virtual void OnLowMemory();

	virtual void OnBatteryLevelChanged(Tizen::System::BatteryLevel batteryLevel);
	virtual void OnScreenOn();
	virtual void OnScreenOff();

	virtual void OnTimerExpired(Tizen::Base::Runtime::Timer& timer);

public:
	class TizenForm* __pForm;
	Tizen::Base::Runtime::Timer* __pTimer;
	AppThread mAppThread;
};



class TizenForm :
	public Tizen::Ui::Controls::Form,
	public Tizen::Ui::ITouchEventListener,
	public Tizen::Ui::IKeyEventListener
{
public:
	TizenForm(TizenApp* pApp);
	
	virtual ~TizenForm();
	
	virtual result Init();
	virtual result OnInitializing();

	virtual result OnDraw();
		
	virtual void OnTouchPressed(const Tizen::Ui::Control& source, const Tizen::Graphics::Point& currentPosition, const Tizen::Ui::TouchEventInfo & touchInfo);
	virtual void OnTouchLongPressed(const Tizen::Ui::Control& source, const Tizen::Graphics::Point& currentPosition, const Tizen::Ui::TouchEventInfo& touchInfo);
	virtual void OnTouchReleased(const Tizen::Ui::Control& source, const Tizen::Graphics::Point& currentPosition, const Tizen::Ui::TouchEventInfo& touchInfo);
	virtual void OnTouchMoved(const Tizen::Ui::Control& source, const Tizen::Graphics::Point& currentPosition, const Tizen::Ui::TouchEventInfo& touchInfo);
	virtual void OnTouchDoublePressed(const Tizen::Ui::Control& source, const Tizen::Graphics::Point& currentPosition, const Tizen::Ui::TouchEventInfo& touchInfo);
	virtual void OnTouchFocusIn(const Tizen::Ui::Control& source, const Tizen::Graphics::Point& currentPosition, const Tizen::Ui::TouchEventInfo& touchInfo);
	virtual void OnTouchFocusOut(const Tizen::Ui::Control& source, const Tizen::Graphics::Point& currentPosition, const Tizen::Ui::TouchEventInfo& touchInfo);
	virtual void OnTouchCanceled(const Tizen::Ui::Control& source, const Tizen::Graphics::Point& currentPosition, const Tizen::Ui::TouchEventInfo& touchInfo);

	virtual void OnKeyPressed(const Tizen::Ui::Control& source, Tizen::Ui::KeyCode keyCode);
	virtual void OnKeyReleased(const Tizen::Ui::Control& source, Tizen::Ui::KeyCode keyCode);
	virtual void OnKeyLongPressed(const Tizen::Ui::Control& source, Tizen::Ui::KeyCode keyCode);

protected:
	TizenApp* __pApp;
	int width, height;

	Tizen::Graphics::Bitmap* m_SplashScreen;
};

#endif

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
