/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_PROCESS_CONTEXT_H
#define HKNP_PROCESS_CONTEXT_H

#include <Common/Visualize/hkProcessContext.h>
#include <Physics/Physics/Extensions/Viewers/hknpViewerColorScheme.h>

class hknpWorld;
class hkProcess;

#define HKNP_PROCESS_CONTEXT_TYPE_STRING "Physics"


/// A simple interface that viewers that want to know when hknpWorlds are added and removed from the physics context
/// can implement.
class hknpProcessContextListener
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpProcessContextListener );

		virtual ~hknpProcessContextListener() {}

		virtual void worldAddedCallback( hknpWorld* newWorld ) = 0;

		virtual void worldRemovedCallback( hknpWorld* newWorld ) = 0;

		virtual hkProcess* getProcess() { return HK_NULL; }
};


/// This is the context that processes (viewers) can use if they understand physics worlds.
/// It listens on all added worlds and can trigger deletion and addition callbacks to processes (viewers) if requested.
/// A context itself is just a data store for any information that a viewer will need. In this case it is pointers to
/// hknpWorlds, as from that any of the physics viewers can find the parts they are interested in and then register
/// themselves with the world to get the appropriate callbacks to keep their state up to date.
class hknpProcessContext : public hkReferencedObject, public hkProcessContext
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_PHYSICS);

		/// Constructor.
		hknpProcessContext();

		/// Register all known Physics processes (viewers). If you don't call this or a subset of what it calls,
		/// all you will get is the common viewers in hkVisualize (debug display and stats).
		static void HK_CALL registerAllProcesses();

		/// Set the color scheme. Use NULL to apply the default color scheme.
		void setColorScheme( hknpViewerColorScheme* colorScheme );

		/// Get the color scheme.
		inline hknpViewerColorScheme* getColorScheme() { return m_colorScheme; }

		/// Register a world. This context listens to world deletion signals, and will remove the world when it sees it
		/// deleted. It does not reference the world, just keeps track of it by way of the listeners.
		void addWorld( hknpWorld* newWorld );

		/// Explicitly remove a world from the context. If you delete a world it will remove itself from the context
		/// anyway as this context listens for world deletion signals.
		void removeWorld( hknpWorld* newWorld );

		/// Find the index of the given world. Returns -1 if not found in this context.
		int findWorld( hknpWorld* world );

		/// Get number of worlds registered in this context.
		inline int getNumWorlds() { return m_worlds.getSize(); }

		/// Get the i-th world registered in this context.
		inline hknpWorld* getWorld( int i ) { return m_worlds[i]; }

		/// So that processes can see all the worlds dynamically, they can be hknpProcessContextWorldListener which
		/// simply get world added to this context events. As such they would be able to then register themselves as
		/// rigid body added listeners and so on and for instance be able to create bodies to display upon those further
		/// callbacks.
		void addWorldListener( hknpProcessContextListener* listener );

		/// So that processes can see all the worlds dynamically, they can be hknpProcessContextWorldListener which
		/// simply can get world removed from this context, and can update their state accordingly
		/// (remove some display geometries etc).
		void removeWorldListener( hknpProcessContextListener* listener );

		/// Returns the array of listeners
		HK_FORCE_INLINE const hkArray<hknpProcessContextListener*>& getWorldListeners() const	{ return m_addListeners; }
		HK_FORCE_INLINE hkArray<hknpProcessContextListener*>& accessWorldListeners()			{ return m_addListeners; }

		/// Signal handler for world deletions.
		void onWorldDestroyedSignal( hknpWorld* world );

		//
		// hkProcessContext implementation
		//

		/// As there can be any number of different user types of data contexts, the type is simply identified by string.
		virtual const char* getType() { return HKNP_PROCESS_CONTEXT_TYPE_STRING; }

		/// Set the VDB that owns this context. This is called by the VDB itself.
		virtual void setOwner( hkVisualDebugger* vdb );

	protected:

		/// As a context must exist at least as long as the VDB, we explicitly do not allow local variables of it to
		/// force the use of 'new' and removeReference().
		/// The VDB itself can't add a reference a context is just a abstract interface that any user data item can
		/// implement for their own viewers.
		virtual ~hknpProcessContext();

		/// Iterates through existing physics objects and adds them for inspection (tweaking) and then
		/// adds listeners to pick up when that state changes and objects are added or removed etc.
		/// Called upon addWorld().
		void addForInspection( hknpWorld* world );

		/// Iterates through existing physics objects and removes them for inspection (tweaking).
		/// Called upon removeWorld().
		void removeFromInspection( hknpWorld* world );

	protected:

		hkArray<hknpWorld*> m_worlds;
		hkArray<hknpProcessContextListener*> m_addListeners;

		// Color scheme
		hknpViewerColorScheme*			m_colorScheme;
		hknpDefaultViewerColorScheme	m_defaultColorScheme;
};


#endif // HKNP_PROCESS_CONTEXT_H

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
