/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_VEHICLE_TYPEMARKS_INFO_H
#define HKNP_VEHICLE_TYPEMARKS_INFO_H

extern const class hkClass hknpTyremarksWheelClass;
extern const class hkClass hknpTyremarkPointClass;
extern const class hkClass hknpTyremarksInfoClass;

class hknpVehicleInstance;
class hknpVehicleData;


/// A tyremark point is defined by two points (left and right and the strength of
/// the tyremark). Having two points instead of one allows for thickness in
/// tyremarks. The strength is a user value that can be used, for example, to
/// shade tyremarks depending on the amount of skidding
struct hknpTyremarkPoint
{
	public:

		HK_DECLARE_REFLECTION();

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_VEHICLE, hknpTyremarkPoint);

			/// Default constructor
		hknpTyremarkPoint();

			/// Serialization constructor.
		hknpTyremarkPoint(hkFinishLoadedObjectFlag f) {}

			/// The strength of a tyremarkPoint is stored in the w-component of the vectors.
			/// The strength is in the range 0.0f to 255.0f.
		hkReal getTyremarkStrength() const;

	public:

			/// The left position of the tyremark.
		hkVector4 m_pointLeft;

			/// The right position of the tyremark.
		hkVector4 m_pointRight;
};


/// hknpTyremarksWheel stores a list of tyremarks associated with a particular wheel.
/// This is a circular array, so old tyremarks eventually get replaced by new tyremarks.
class hknpTyremarksWheel : public hkReferencedObject
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_VEHICLE);
		HK_DECLARE_REFLECTION();

			/// Default constructor
		hknpTyremarksWheel();

			/// Serialization constrcutor.
		hknpTyremarksWheel(hkFinishLoadedObjectFlag f) : hkReferencedObject(f), m_tyremarkPoints(f) {}

			/// Destructor.
		virtual ~hknpTyremarksWheel() {}

			/// Sets the number of tyre mark points that can be stored.
		void setNumPoints(int num_points);

			/// Add a tyreMarkPoint to the array.
		void addTyremarkPoint( hknpTyremarkPoint& point);

			/// Returns the i-th stored tyremark point in the object.
		const hknpTyremarkPoint& getTyremarkPoint(int point) const;

	public:

			/// Current position in the array of tyremarkPoints.
		int m_currentPosition;

			/// The number of points in the array
		int m_numPoints;

			/// Circular array of tyreMarkPoints.
		hkArray<struct hknpTyremarkPoint> m_tyremarkPoints;
};


/// hknpTyremarksInfo stores a list of hknpTyremarksWheel for a particular vehicle
class hknpTyremarksInfo : public hkReferencedObject
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_VEHICLE);
		HK_DECLARE_REFLECTION();

		/// Constructor. Takes a hknpVehicleData object and the number of skidmark points to store.
		hknpTyremarksInfo(const hknpVehicleData& data, int num_points);

			/// Serialization constructor.
		hknpTyremarksInfo(hkFinishLoadedObjectFlag f) : hkReferencedObject(f), m_tyremarksWheel(f) {}

			/// Destructor.
		virtual ~hknpTyremarksInfo();

			/// Updates Tyremark information
		virtual void updateTyremarksInfo(hkReal timestep, const hknpVehicleInstance* vehicle);

			/// Retrieves the Tyremark information in the form of a strip
		virtual void getWheelTyremarksStrips(const hknpVehicleInstance* vehicle, int wheel, hkVector4* strips_out) const;

	public:

			/// The minimum energy a tyremark should have. The actual strength of a point will
			/// be scaled to be between 0.0f and 255.0f.
		hkReal m_minTyremarkEnergy;

			/// The maximum energy a tyremark should have. The actual strength of a point will
			/// be scaled to be between 0.0f and 255.0f.
		hkReal m_maxTyremarkEnergy;

			/// There is a hknpTyremarksWheel for each wheel.
		hkArray<class hknpTyremarksWheel*> m_tyremarksWheel;
};

#endif // HKNP_VEHICLE_TYPEMARKS_INFO_H

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
