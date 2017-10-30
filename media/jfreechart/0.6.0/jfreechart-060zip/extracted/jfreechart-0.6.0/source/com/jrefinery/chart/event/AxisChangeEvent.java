/* =======================================
 * JFreeChart : a Java Chart Class Library
 * =======================================
 *
 * Project Info:  http://www.jrefinery.com/jfreechart
 * Project Lead:  David Gilbert (david.gilbert@jrefinery.com);
 *
 * This file...
 * $Id: AxisChangeEvent.java,v 1.3 2001/11/12 09:02:31 mungady Exp $
 *
 * Original Author:  David Gilbert;
 * Contributor(s):   -;
 *
 * (C) Copyright 2000, 2001, Simba Management Limited;
 *
 * This library is free software; you can redistribute it and/or modify it under the terms
 * of the GNU Lesser General Public License as published by the Free Software Foundation;
 * either version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with this library;
 * if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA 02111-1307, USA.
 *
 * Changes (from 24-Aug-2001)
 * --------------------------
 * 24-Aug-2001 : Added standard source header. Fixed DOS encoding problem (DG);
 * 07-Nov-2001 : Updated e-mail address in header (DG);
 *
 */

package com.jrefinery.chart.event;

import com.jrefinery.chart.*;

/**
 * A change event that encapsulates information about a change to an axis.
 */
public class AxisChangeEvent extends ChartChangeEvent {

    /** The axis that generated the change event. */
    protected Axis axis;

    /**
     * Default constructor: returns a new AxisChangeEvent.
     * @param axis The axis that generated the event.
     */
    public AxisChangeEvent(Axis axis) {
        super(axis);
        this.axis = axis;
    }

    /**
     * Returns a reference to the axis that generated the event.
     */
    public Axis getAxis() {
        return axis;
    }

}
